"""Versioned fictional food-delivery simulation; never tuned to model scores."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import FEATURES as AUTHOR_FEATURES

FEATURES = AUTHOR_FEATURES + ['text_length', 'hashtag_count', 'mention_count', 'has_media',
    'has_url', 'first_hour_impressions', 'first_hour_likes', 'first_hour_replies', 'first_hour_retweets']
BOUNDS = {'hour_utc': [0, 23], 'weekday': [0, 6], **{name: [0, 1] for name in
    ['verified', 'is_reply', 'is_quote', 'has_media', 'has_url']}}
PARAMETERS = {
    'in_distribution': dict(reach_scale=1.0, media_effect=.35, late_noise=.85, amplification_sigma=.8, shock_probability=.015, shock_scale=60),
    'shifted': dict(reach_scale=.65, media_effect=.05, late_noise=1.2, amplification_sigma=1.15, shock_probability=.04, shock_scale=95),
}
TOPICS = {
 'late_delivery': ('My {food} from {shop} is {minutes} minutes late.', 'Dinner from {shop} arrived on time today.'),
 'missing_item': ('The {food} was missing from my {shop} order.', 'Every item in my {shop} order arrived correctly.'),
 'food_quality': ('My {food} from {shop} arrived cold and soggy.', 'The {food} from {shop} was fresh and delicious.'),
 'refund': ('Still waiting for my ₹{amount} refund after a cancelled {shop} order.', 'My ₹{amount} refund from the cancelled order came through.'),
 'support': ('Support keeps repeating the same answer about my {shop} order.', 'Support sorted out my {shop} order quickly.'),
 'app_payment': ('The app charged ₹{amount} but my {shop} order failed.', 'Payment and checkout for {shop} were smooth today.'),
 'rider_experience': ('The delivery handoff for my {shop} order was confusing.', 'Thanks to the rider who delivered my {food} in the rain.'),
 'food_safety': ('I found a piece of plastic in my {food} from {shop}. Please help.', 'The packaging and seal on my {shop} order looked intact.'),
}
LIMITATIONS = [
 'All posts, authors, engagement and reference labels are fictional simulation outputs, not observed customer data.',
 'Scores measure recovery of assumptions in this simulator, not performance on real social media or Indian customers.',
 'Prediction is made after the first hour; it classifies eventual simulated spread, not a forecast of first-hour engagement.',
 'Independent latent appeal, platform amplification, late noise and rare shocks remain unobserved by ML.',
 'The shifted test is one specified stress scenario, not comprehensive evidence of real-world robustness.',
 'Reference sentiment/topic labels come from templates and are not human-annotated ground truth.',
 'The 5:1 missed-positive to false-alert cost is a prototype assumption; all responses still need human review.',
]


def simulate(n=15000, seed=20261029, shifted=False):
    rng = np.random.default_rng(seed)
    regime = 'shifted' if shifted else 'in_distribution'
    p = PARAMETERS[regime]
    prefix = 'ood' if shifted else 'syn'
    followers = np.minimum(rng.lognormal(5.7, 1.75, n), 2_000_000).astype(int)
    media = rng.binomial(1, .32 if not shifted else .5, n)
    appeal = rng.normal(0, 1, n)
    early_noise = rng.lognormal(-.25, .75, n)
    reach = p['reach_scale'] * (4 + 1.5 * (followers + 1)**.66) * np.exp(.45*appeal + p['media_effect']*media) * early_noise
    impressions = rng.poisson(np.minimum(reach, 5_000_000))
    likes = rng.binomial(impressions, np.clip(.035*np.exp(.25*appeal), .001, .4))
    replies = rng.binomial(impressions, np.clip(.004*np.exp(.2*appeal), .0001, .08))
    early_rt = rng.binomial(impressions, np.clip(.008*np.exp(.3*appeal), .0001, .15))
    amplification = rng.lognormal(-p['amplification_sigma']**2/2, p['amplification_sigma'], n)
    late_noise = np.exp(np.clip(rng.standard_t(4, n)*p['late_noise'], -5, 6))
    shock = rng.random(n) < p['shock_probability']
    shock_multiplier = np.where(shock, 1+p['shock_scale']*(1+rng.pareto(2.2, n)), 1)
    late_mean = (2 + 1.8*early_rt + .25*(likes+1)**.8 + .15*(followers+1)**.35)
    late_mean *= np.exp(.5*appeal + p['media_effect']*media) * amplification * late_noise * shock_multiplier
    # Gamma-Poisson mixture adds irreducible count noise; cap is only a numerical safeguard.
    later = rng.poisson(np.minimum(rng.gamma(2, np.minimum(late_mean, 1e7)/2), 1e7))
    date = pd.Timestamp('2026-01-01', tz='UTC') + pd.to_timedelta(rng.integers(0, 90*86400, n), unit='s')
    frame = pd.DataFrame(dict(post_key=[f'{prefix}-post-{i:06}' for i in range(n)],
        user_key=[f'{prefix}-author-{i:06}' for i in range(n)], date=date,
        followers_count=followers, friends_count=rng.lognormal(5.4, .9, n).astype(int),
        listed_count=rng.poisson(np.minimum(followers/300, 10000)), statuses_count=rng.lognormal(7, 1.4, n).astype(int),
        verified=rng.binomial(1, np.clip(.01+followers/300000, .01, .85)),
        account_age_days=rng.integers(30, 6000, n), hour_utc=date.hour, weekday=date.dayofweek,
        is_reply=rng.binomial(1, .25, n), is_quote=rng.binomial(1, .08, n), has_media=media,
        first_hour_impressions=impressions, first_hour_likes=likes, first_hour_replies=replies,
        first_hour_retweets=early_rt, retweet_count=early_rt+later, synthetic=True, simulation_regime=regime))
    texts, topics, sentiments = [], [], []
    for _ in range(n):
        topic = str(rng.choice(list(TOPICS)))
        sentiment = str(rng.choice(['negative', 'positive', 'neutral'], p=[.55, .3, .15]))
        shop = str(rng.choice(['Fictional Spice Corner', 'Fictional Bento Box', 'Fictional Dosa House', 'Fictional Pizza Lane']))
        food = str(rng.choice(['biryani', 'dosa', 'pizza', 'noodles', 'paneer wrap', 'rice bowl']))
        text = ('Checking the status of my {food} order from {shop}.' if sentiment == 'neutral' else TOPICS[topic][sentiment == 'positive'])
        text = text.format(food=food, shop=shop, amount=int(rng.integers(100, 1500)), minutes=int(rng.integers(15, 100)))
        text += str(rng.choice(['', ' This was my lunch break.', ' Ordered for a family dinner.', ' Can someone at Tomato take a look?']))
        if rng.random() < .3:
            text += ' #TomatoDelivery'
        if rng.random() < .65:
            text = '@TomatoHelp ' + text
        if rng.random() < .08:
            text += ' https://example.invalid/order'
        texts.append(text); topics.append(topic); sentiments.append(sentiment)
    frame['text'], frame['reference_topic'], frame['reference_sentiment'] = texts, topics, sentiments
    frame['text_length'] = frame.text.str.len()
    frame['hashtag_count'] = frame.text.str.count('#')
    frame['mention_count'] = frame.text.str.count('@')
    frame['has_url'] = frame.text.str.contains('https://', regex=False).astype(int)
    # Hidden variables are written to an isolated audit file, never to feature preparation.
    latent = pd.DataFrame(dict(post_key=frame.post_key, latent_appeal=appeal,
        platform_amplification=amplification, late_noise=late_noise, shock=shock, shock_multiplier=shock_multiplier))
    return frame.sort_values(['date', 'post_key']).reset_index(drop=True), latent


def generate(directory='data/synthetic', n=15000, ood_n=3000, seed=20261029):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame, latent = simulate(n, seed)
    ood, ood_latent = simulate(ood_n, seed+1, shifted=True)
    for name, data, audit in [('food_delivery', frame, latent), ('shifted_test', ood, ood_latent)]:
        data.to_json(directory/f'{name}.jsonl', orient='records', lines=True, date_format='iso', force_ascii=False)
        audit.to_csv(directory/f'{name}_latent_audit.csv', index=False)
    (directory/'simulation_manifest.json').write_text(json.dumps(dict(version=1, seed=seed, ood_seed=seed+1,
        rows=n, shifted_rows=ood_n, parameters=PARAMETERS, features=FEATURES, limitations=LIMITATIONS), indent=2), encoding='utf-8')
    return frame, ood
