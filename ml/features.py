"""Fold-local numerical feature engineering from the prediction-time whitelist."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ml.synthetic import FEATURES


class EngagementFeatures(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        self.n_features_in_ = np.asarray(x).shape[1]
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        values = {name: x[:, i] for i, name in enumerate(FEATURES)}
        impressions = values['first_hour_impressions']
        followers = values['followers_count']
        likes, replies, retweets = [values['first_hour_'+name] for name in ['likes', 'replies', 'retweets']]
        total = likes+replies+retweets
        extra = [likes/(impressions+10), replies/(impressions+10), retweets/(impressions+10),
                 total/(impressions+10), np.log1p(impressions/(followers+10)),
                 np.log1p(retweets/(followers+10)), retweets/(total+3), replies/(total+3),
                 np.log1p(values['listed_count']/(followers+10)),
                 np.sin(2*np.pi*values['hour_utc']/24), np.cos(2*np.pi*values['hour_utc']/24),
                 np.sin(2*np.pi*values['weekday']/7), np.cos(2*np.pi*values['weekday']/7)]
        return np.column_stack([np.log1p(np.maximum(x, 0))]+extra)


class EngagementFeaturesV15(EngagementFeatures):
    """Additional prediction-time ratios; legacy transformer stays unchanged."""
    def transform(self, x):
        x=np.asarray(x,dtype=float)
        v={name:x[:,index] for index,name in enumerate(FEATURES)}
        likes,replies,retweets=[v['first_hour_'+name] for name in ['likes','replies','retweets']]
        total=likes+replies+retweets
        extra=[np.log1p(total/(v['followers_count']+10)),np.log1p(likes/(retweets+1)),
               np.log1p(replies/(likes+1)),np.log1p(v['followers_count']/(v['friends_count']+1)),
               np.log1p(v['statuses_count']/(v['account_age_days']+1))]
        return np.column_stack([super().transform(x)]+extra)
