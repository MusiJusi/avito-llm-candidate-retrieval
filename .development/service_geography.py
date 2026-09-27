"""Service-specific location priors, computed only from isolated history.

Remote services and on-site services should not share one locality assumption.
Sparse search-location/service counts shrink to a service locality prior. Unknown
services/cities remain explicit missing evidence, never a hard location filter.
"""
import numpy as np

SERVICE_GEO_FEATURE_NAMES=['service_same_location_probability','service_location_transition_probability',
                           'service_log_support','service_local_log_support','service_geo_supported']


class ServiceGeography:
    def __init__(self,history,shrinkage=20.):
        self.shrinkage=shrinkage
        same=history.search_location_id.eq(history.item_location_id)
        self.global_same=float(same.mean())
        self.service_count=history.groupby('item_microcat_id').size().to_dict()
        self.service_same=history.assign(_same=same.astype(float)).groupby('item_microcat_id')._same.sum().to_dict()
        self.location_count=history.groupby('search_location_id').size().to_dict()
        self.global_locations=(history.groupby('item_location_id').size()/len(history)).to_dict()
        self.location_pair=history.groupby(['search_location_id','item_location_id']).size().to_dict()
        self.service_local=history.groupby(['search_location_id','item_microcat_id']).size().to_dict()
        self.service_pair=history.groupby(['search_location_id','item_microcat_id','item_location_id']).size().to_dict()

    def features(self,search_location,microcats,item_locations):
        alpha=self.shrinkage
        location_total=self.location_count.get(search_location,0)
        base_self=(self.location_pair.get((search_location,search_location),0)+alpha*self.global_locations.get(search_location,0))/(location_total+alpha)
        rows=[]
        for micro,location in zip(microcats,item_locations):
            count=self.service_count.get(int(micro),0)
            local=self.service_local.get((search_location,int(micro)),0)
            if not count:
                rows.append([np.nan,np.nan,0.,0.,0.]);continue
            same=(self.service_same.get(int(micro),0)+alpha*self.global_same)/(count+alpha)
            if location not in self.global_locations:
                probability=np.nan
            else:
                base=(self.location_pair.get((search_location,int(location)),0)+alpha*self.global_locations[int(location)])/(location_total+alpha)
                prior=same if location==search_location else (1-same)*base/max(1-base_self,1e-12)
                probability=(self.service_pair.get((search_location,int(micro),int(location)),0)+alpha*prior)/(local+alpha)
            rows.append([same,probability,np.log1p(count),np.log1p(local),1.])
        result=np.asarray(rows,np.float32)
        assert result.shape==(len(microcats),len(SERVICE_GEO_FEATURE_NAMES)) and not np.isinf(result).any()
        return result
