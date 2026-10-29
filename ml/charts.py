"""Plain-array Plotly figures compatible with older Streamlit frontends."""
import numpy as np
import plotly.graph_objects as go

BLUE='#4da6ff'
ORANGE='#ffad66'


def layout(fig,title,x,y):
    fig.update_layout(title=title,xaxis_title=x,yaxis_title=y,height=410,
        margin=dict(l=40,r=20,t=65,b=90),template='plotly_white',font=dict(size=12),
        title_font=dict(size=15),legend=dict(orientation='h',x=0,y=-.25,xanchor='left',yanchor='top'))
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def confusion(values,title='Final test confusion matrix'):
    z=[[int(values['tn']),int(values['fp'])],[int(values['fn']),int(values['tp'])]]
    text=[[f'TN = {z[0][0]}',f'FP = {z[0][1]}'],[f'FN = {z[1][0]}',f'TP = {z[1][1]}']]
    fig=go.Figure(go.Heatmap(z=z,x=['Ordinary','High spread'],y=['Ordinary','High spread'],
        text=text,texttemplate='%{text}',colorscale='Blues',showscale=False,
        hovertemplate='Actual %{y}<br>Predicted %{x}<br>%{text}<extra></extra>'))
    fig.update_yaxes(autorange='reversed')
    return layout(fig,title,'Predicted class','Actual class')


def categorical_confusion(matrix,title):
    if matrix.empty or matrix.isna().any().any():
        raise ValueError('Confusion matrix is empty or contains missing counts.')
    counts=matrix.to_numpy(dtype=int)
    if (counts<0).any():
        raise ValueError('Confusion counts must be nonnegative.')
    fig=go.Figure(go.Heatmap(z=counts.tolist(),x=matrix.columns.astype(str).tolist(),
        y=matrix.index.astype(str).tolist(),text=counts.astype(str).tolist(),texttemplate='%{text}',
        colorscale='Blues',showscale=False,
        hovertemplate='Reference %{y}<br>Predicted %{x}<br>Posts %{z}<extra></extra>'))
    fig.update_yaxes(autorange='reversed')
    return layout(fig,title,'Predicted class','Reference class')


def precision_recall(data,operating,prevalence):
    fig=go.Figure(go.Scatter(x=data.recall.astype(float).tolist(),y=data.precision.astype(float).tolist(),
        mode='lines',name='Final model',line=dict(color=BLUE,width=2)))
    fig.add_trace(go.Scatter(x=[float(operating['recall'])],y=[float(operating['precision'])],
        mode='markers',name='Frozen operating point',marker=dict(color=ORANGE,size=13)))
    fig.add_hline(y=float(prevalence),line_dash='dash',annotation_text='Class prevalence')
    return layout(fig,'Final test precision–recall curve','Recall','Precision')


def roc(data):
    fig=go.Figure(go.Scatter(x=data.false_positive_rate.astype(float).tolist(),y=data.true_positive_rate.astype(float).tolist(),
        mode='lines',name='Final model',line=dict(color=BLUE,width=2)))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Chance',line=dict(dash='dash',color='gray')))
    return layout(fig,'Final test ROC curve','False positive rate','True positive rate')


def importance(data):
    data=data.sort_values('importance').tail(12)
    names={'first_hour_retweets':'1h retweets','first_hour_likes':'1h likes','first_hour_impressions':'1h impressions',
        'followers_count':'Followers','friends_count':'Following','statuses_count':'Lifetime posts','listed_count':'List memberships',
        'account_age_days':'Account age','hashtag_count':'Hashtags','mention_count':'Mentions','text_length':'Post length'}
    labels=[names.get(value,value.replace('_',' ').title()) for value in data.feature.astype(str)]
    fig=go.Figure(go.Bar(x=data.importance.astype(float).tolist(),y=labels,
        orientation='h',marker_color=BLUE,error_x=dict(type='data',array=data['std'].astype(float).tolist())))
    fig.update_layout(margin=dict(l=175,r=25,t=65,b=50))
    return layout(fig,'Importance · validation permutation','Decrease in AP','Input feature')


def calibration(data):
    fig=go.Figure(go.Scatter(x=data.mean_probability.astype(float).tolist(),y=data.fraction_positive.astype(float).tolist(),
        mode='lines+markers',name='Observed',marker=dict(color=BLUE,size=8),customdata=data['count'].astype(int).tolist(),
        hovertemplate='Mean probability %{x:.3f}<br>Observed positive fraction %{y:.3f}<br>n=%{customdata}<extra></extra>'))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Perfect calibration',line=dict(dash='dash',color='gray')))
    return layout(fig,'Final test probability calibration','Mean predicted probability','Observed high-spread fraction')


def probability_distribution(data):
    fig=go.Figure()
    for label,name,color in [(0,'Actual ordinary',BLUE),(1,'Actual high spread',ORANGE)]:
        values=data.loc[data.label==label,'probability'].astype(float).tolist()
        fig.add_trace(go.Histogram(x=values,name=name,marker_color=color,opacity=.65,
            xbins=dict(start=0,end=1,size=.05),histnorm='probability'))
    fig.update_layout(barmode='overlay')
    return layout(fig,'Predicted probabilities<br>by actual test class','Predicted probability','Fraction within each class')


def build_figures(bundle):
    result=bundle['metrics']['test']['operational']
    return [confusion(result),precision_recall(bundle['pr'],result,result['prevalence']),roc(bundle['roc']),
        importance(bundle['importance']),calibration(bundle['calibration'].query("split == 'test'")),
        probability_distribution(bundle['predictions'].query("split == 'test'"))]
