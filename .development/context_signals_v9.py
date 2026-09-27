"""Independent history features: uncertain geography and semantic transfer.

This module consumes an already isolated history. It never receives query gold,
benchmark answers, search_category or a query ID. Aggregates cannot recover a
held positive unless that interaction was mistakenly admitted by the caller.
The general ideas are historical coordinate imputation and nearest-neighbour
regression; no implementation or fitted artifacts from another solution are used.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

GEO_NAMES = ['context_log_distance', 'context_center_source', 'context_log_support',
             'context_log_spread50', 'context_log_spread90', 'context_geo_confidence',
             'context_same_location_share', 'context_peak_location_share',
             'context_location_entropy', 'context_geo_available']
NEIGHBOR_NAMES = ['neighbor_centroid_cosine', 'neighbor_log_centroid_rank',
                  'neighbor_micro_probability', 'neighbor_micro_relative',
                  'neighbor_micro_entropy', 'neighbor_micro_coverage',
                  'neighbor_top_similarity', 'neighbor_similarity_margin',
                  'neighbor_effective_count']

def kilometers(lat, lon, center):
    """Vectorized great-circle distance; missing coordinates remain missing."""
    latitude=np.radians(np.asarray(lat,float));longitude=np.radians(np.asarray(lon,float))
    c_lat,c_lon=np.radians(center)
    a=np.sin((latitude-c_lat)/2)**2+np.cos(latitude)*np.cos(c_lat)*np.sin((longitude-c_lon)/2)**2
    return 12742*np.arcsin(np.sqrt(np.clip(a,0,1)))

class GeographicEvidence:
    def __init__(self,history,coordinates,corpus_centers):
        self.corpus_centers={int(k):np.asarray(v,float) for k,v in corpus_centers.items()
                             if np.isfinite(v).all()}
        # Distinct text/context-item observations already come from the caller's
        # cleaned history. Read coordinate metadata separately to keep long texts
        # out of RAM, and join only admitted interaction rows.
        located=history[['search_location_id','item_id']].join(coordinates,on='item_id').dropna()
        self.centers={int(k):v for k,v in located.groupby('search_location_id')[['lat','lon']].median().iterrows()}
        self.support=located.groupby('search_location_id').size().to_dict()
        if len(located):
            aligned=np.stack([self.centers[int(k)].to_numpy() for k in located.search_location_id])
            latitude=np.radians(located.lat.to_numpy());longitude=np.radians(located.lon.to_numpy())
            center_lat,center_lon=np.radians(aligned).T
            a=np.sin((latitude-center_lat)/2)**2+np.cos(latitude)*np.cos(center_lat)*np.sin((longitude-center_lon)/2)**2
            located=located.assign(distance=12742*np.arcsin(np.sqrt(np.clip(a,0,1))))
            self.spread50=located.groupby('search_location_id').distance.quantile(.5).to_dict()
            self.spread90=located.groupby('search_location_id').distance.quantile(.9).to_dict()
        else:self.spread50={};self.spread90={}
        counts=history.groupby(['search_location_id','item_location_id']).size()
        self.location_stats={}
        for source,group in counts.groupby(level=0):
            probability=group.to_numpy(float)/group.sum()
            same=float(group.get((source,source),0)/group.sum())
            entropy=float(-(probability*np.log(np.maximum(probability,1e-12))).sum())
            self.location_stats[int(source)]=(same,float(probability.max()),entropy)

    def evidence(self,location):
        location=int(location)
        historical=self.centers.get(location)
        support=float(self.support.get(location,0))
        spread=float(self.spread50.get(location,np.nan))
        if location in self.corpus_centers:center=self.corpus_centers[location];source=0.
        elif historical is not None:center=historical.to_numpy();source=1.
        else:center=None;source=2.
        # Confidence is continuous, so neither a tiny sample nor a broad regional
        # search is silently treated as an exact city coordinate.
        confidence=support/(support+30)*np.exp(-spread/80) if np.isfinite(spread) else 0.
        stats=self.location_stats.get(location,(np.nan,np.nan,np.nan))
        constants=[source,np.log1p(support),np.log1p(spread),
                   np.log1p(self.spread90.get(location,np.nan)),confidence,*stats,float(center is not None)]
        return center,np.asarray(constants,np.float32)

    def features(self,location,latitudes,longitudes):
        center,constants=self.evidence(location)
        distance=kilometers(latitudes,longitudes,center) if center is not None else np.full(len(latitudes),np.nan)
        return np.column_stack([np.log1p(distance),np.tile(constants,(len(distance),1))]).astype(np.float32)

    def additional_candidates(self,location,teacher,learned,item_latitudes,item_longitudes,topk):
        center,constants=self.evidence(location)
        if constants[0]!=1 or constants[4]<.1:return np.empty(0,np.int32)
        distance=kilometers(item_latitudes,item_longitudes,center)
        scale=max(35.,float(self.spread50.get(int(location),35.)))
        proximity=np.nan_to_num(np.exp(-distance/scale),nan=0.)
        # A separate list preserves the old geographic feature distribution. The
        # ranker receives distance/confidence explicitly instead of a hard filter.
        return np.union1d(topk(teacher+.12*proximity,128),topk(learned+.12*proximity,128))

class SemanticEvidence:
    def __init__(self,history,text_vector_lookup,document_ids,document_vectors,microcats):
        admitted=history[['query_norm','item_id','item_microcat_id']].drop_duplicates(['query_norm','item_id'])
        self.texts=sorted(admitted.query_norm.unique())
        text_rows={text:i for i,text in enumerate(self.texts)}
        self.vectors=np.stack([text_vector_lookup[text] for text in self.texts]).astype(np.float32)
        document_rows={item:i for i,item in enumerate(document_ids)}
        rows=admitted.query_norm.map(text_rows).to_numpy()
        columns=admitted.item_id.map(document_rows).to_numpy()
        assert np.isfinite(columns).all(),'Every admitted item requires a frozen document vector'
        incidence=sp.csr_matrix((np.ones(len(rows),np.float32),(rows,columns)),
                                shape=(len(self.texts),len(document_ids)))
        incidence=sp.diags(1/np.maximum(np.asarray(incidence.sum(axis=1)).ravel(),1))@incidence
        self.centroids=np.asarray(incidence@document_vectors,dtype=np.float32)
        classes={int(m):i for i,m in enumerate(microcats)}
        valid=admitted.item_microcat_id.isin(classes)
        counts=sp.csr_matrix((np.ones(valid.sum(),np.float32),
            (rows[valid],admitted.loc[valid,'item_microcat_id'].map(classes))),shape=(len(self.texts),len(classes)))
        # The total count includes classes outside the searchable corpus. Coverage
        # is retained as a feature rather than silently renormalizing it away.
        denominator=admitted.groupby('query_norm').size().reindex(self.texts).to_numpy()
        self.distributions=sp.diags(1/np.maximum(denominator,1))@counts
        self.microcats=np.asarray(microcats)

    def query_evidence(self,vectors,neighbors=24):
        vectors=np.asarray(vectors,np.float32)
        prototypes=[];probabilities=[];statistics=[]
        matrix=self.vectors.astype(np.float64).T
        for start in range(0,len(vectors),128):
            similarities=(vectors[start:start+128].astype(np.float64)@matrix).astype(np.float32)
            for similarity in similarities:
                k=min(neighbors,len(similarity))
                nearest=np.argpartition(-similarity,k-1)[:k]
                nearest=nearest[np.lexsort((nearest,-similarity[nearest]))]
                weights=np.exp((similarity[nearest]-similarity[nearest[0]])/.1).astype(np.float64)
                weights/=weights.sum()
                prototype=weights@self.centroids[nearest]
                prototype/=max(np.linalg.norm(prototype),1e-12)
                distribution=np.asarray(self.distributions[nearest].T@weights).ravel()
                coverage=distribution.sum()
                normalized=distribution/max(coverage,1e-12)
                entropy=float(-(normalized*np.log(np.maximum(normalized,1e-12))).sum())
                effective=float(np.exp(-(weights*np.log(np.maximum(weights,1e-12))).sum()))
                margin=float(similarity[nearest[0]]-similarity[nearest[1]]) if k>1 else 0.
                prototypes.append(prototype);probabilities.append(distribution)
                statistics.append([entropy,coverage,float(similarity[nearest[0]]),margin,effective])
        return np.asarray(prototypes,np.float32),np.asarray(probabilities,np.float32),np.asarray(statistics,np.float32)

    @staticmethod
    def features(prototype,probability,statistics,item_vectors,item_micro_rows,rankdata):
        cosine=(item_vectors.astype(np.float64)@prototype.astype(np.float64)).astype(np.float32)
        likelihood=probability[item_micro_rows]
        relative=likelihood/max(float(probability.max()),1e-12)
        return np.column_stack([cosine,np.log1p(rankdata(-cosine,method='min')),likelihood,relative,
            np.tile(statistics,(len(cosine),1))]).astype(np.float32)
