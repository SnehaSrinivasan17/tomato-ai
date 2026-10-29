"""Stream JSONL without retaining customer text or raw user identifiers."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

COUNTS = ['followers_count', 'friends_count', 'listed_count', 'statuses_count']
FEATURES = COUNTS + ['verified', 'account_age_days', 'hour_utc', 'weekday', 'is_reply', 'is_quote']


def count(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return np.nan
    return float(value) if np.isfinite(value) and value >= 0 and value == int(value) else np.nan


def timestamp(value):
    if not isinstance(value, str):
        return pd.NaT
    return pd.to_datetime(value, format='%a %b %d %H:%M:%S %z %Y', errors='coerce', utc=True)


def load_data(path, cohort='originals'):
    if cohort not in {'originals', 'reposts'}:
        raise ValueError('cohort must be originals or reposts')
    audit = Counter()
    rows, seen, seen_sources = [], set(), set()
    observed_fields = set()
    with Path(path).open(encoding='utf-8') as source:
        for line in source:
            audit['input_lines'] += 1
            try:
                post = json.loads(line)
            except json.JSONDecodeError:
                audit['invalid_json'] += 1
                continue
            if not isinstance(post, dict):
                audit['invalid_record'] += 1
                continue
            observed_fields.update(post)
            user = post.get('user')
            if not isinstance(user, dict):
                audit['invalid_user'] += 1
                continue
            post_id, user_id = post.get('id_str', post.get('id')), user.get('id_str', user.get('id'))
            if not str(post_id).isdigit() or not str(user_id).isdigit():
                audit['invalid_identifier'] += 1
                continue
            if str(post_id) in seen:
                audit['duplicate_post'] += 1
                continue
            seen.add(str(post_id))
            repost = 'retweeted_status' in post or str(post.get('text', '')).startswith('RT @')
            if repost:
                audit['reposts_detected'] += 1
            if (cohort == 'originals' and repost) or (cohort == 'reposts' and not repost):
                audit['other_cohort_excluded'] += 1
                continue
            if repost:
                original = post.get('retweeted_status')
                if not isinstance(original, dict):
                    audit['repost_without_original_id'] += 1
                    continue
                original_id = original.get('id_str', original.get('id'))
                if not str(original_id).isdigit():
                    audit['repost_without_original_id'] += 1
                    continue
                if str(original_id) in seen_sources:
                    audit['duplicate_original_source'] += 1
                    continue
                seen_sources.add(str(original_id))
                # Reposts of the same source are not independent examples.
                post_id = original_id
            outcome, date = count(post.get('retweet_count')), timestamp(post.get('created_at'))
            if pd.isna(outcome) or pd.isna(date):
                audit['invalid_target_or_date'] += 1
                continue
            row = {name: count(user.get(name)) for name in COUNTS}
            age = (date - timestamp(user.get('created_at'))).total_seconds() / 86400
            row.update(verified=float(user['verified']) if isinstance(user.get('verified'), bool) else np.nan,
                       account_age_days=age if age >= 0 else np.nan,
                       hour_utc=date.hour, weekday=date.dayofweek,
                       is_reply=float(post.get('in_reply_to_status_id') is not None),
                       is_quote=float(post['is_quote_status']) if isinstance(post.get('is_quote_status'), bool) else np.nan)
            audit['missing_feature_values'] += sum(pd.isna(v) for v in row.values())
            audit['suspicious_counts_over_billion'] += sum(v > 1e9 for v in row.values() if pd.notna(v))
            row.update(post_key=hashlib.sha256(str(post_id).encode()).hexdigest(),
                       user_key=hashlib.sha256(str(user_id).encode()).hexdigest(),
                       date=date, retweet_count=outcome)
            rows.append(row)
    if not rows:
        raise ValueError('No valid records in the requested cohort.')
    frame = pd.DataFrame(rows).sort_values(['date', 'post_key']).reset_index(drop=True)
    audit['eligible_posts'] = len(frame)
    return frame, dict(audit), sorted(observed_fields)


def split_data(frame):
    # Boundaries use timestamps, so simultaneous posts cannot cross a boundary.
    dates = frame.date.sort_values()
    first, second = dates.iloc[int(len(dates) * .6)], dates.iloc[int(len(dates) * .8)]
    train = frame[frame.date < first].copy()
    validation = frame[(frame.date >= first) & (frame.date < second)].copy()
    test = frame[frame.date >= second].copy()
    # Purge users from later partitions to prevent repeated-author leakage.
    validation = validation[~validation.user_key.isin(train.user_key)].copy()
    test = test[~test.user_key.isin(set(train.user_key) | set(validation.user_key))].copy()
    if min(len(train), len(validation), len(test)) < 50:
        raise ValueError('Insufficient user-disjoint chronological data for reliable evaluation.')
    return train, validation, test
