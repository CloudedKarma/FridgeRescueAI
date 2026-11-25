from pathlib import Path
import sqlite3
import json
import streamlit as st
import pandas as pd
from engine import Engine
ROOT=Path(__file__).parent
st.set_page_config(page_title='Fridge Rescue AI')
@st.cache_resource
def load_engine(path, modified, model_modified):
    e=Engine(path)
    if model_modified and path.name=='recipes_foodcom.csv':
        try: e.load_learned(model_path)
        except ValueError as exc: st.warning(str(exc))
    return e
options=[p for p in [ROOT/'recipes_foodcom.csv', ROOT/'recipes.csv'] if p.exists()]
path=st.sidebar.selectbox('Research dataset',options,format_func=lambda p:p.name)
model_path=ROOT/'optimized_results'/'rankers.joblib'
if not model_path.exists(): model_path=ROOT/'ml_results'/'rankers.joblib'
e=load_engine(path,path.stat().st_mtime_ns,model_path.stat().st_mtime_ns if model_path.exists() else 0)
st.title('Fridge Rescue AI')
st.caption(f'Research prototype | {len(e.df):,} recipes | {path.name}. Ingredient extraction may be incomplete. Check original ingredients and quantities before cooking.')
inv=st.multiselect('Ingredients available',sorted(e.vocab))
extra=st.text_input('Other ingredients (comma separated)')
inv+= [x.strip().lower() for x in extra.split(',') if x.strip()]
method=st.selectbox('Recommendation algorithm',['Hybrid','Baseline','TF-IDF','KNN']+ (list(e.learned['models']) + (['FROR (optimized)'] if 'optimized_weights' in e.learned else []) if e.learned else []))
limit=st.slider('Maximum missing ingredients',0,5,2)
def db(sql,args=()):
    with sqlite3.connect(ROOT/'grocery.db') as c:
        c.execute('CREATE TABLE IF NOT EXISTS grocery (item TEXT PRIMARY KEY)')
        return c.execute(sql,args).fetchall()
if inv:
    r=e.rank(inv,method)
    for label,subset in [('All extracted ingredients matched',r[r.missing_count==0]),('Almost available',r[(r.missing_count>0)&(r.missing_count<=limit)])]:
        st.subheader(label)
        if subset.empty: st.write('No matching recipes.')
        for i,row in subset.head(5).iterrows():
            with st.expander(f'{row["name"]} — {row.coverage:.0%} ingredient coverage'):
                st.write('Ingredients:',row.ingredients.replace('|',', ')); st.write('Missing:',', '.join(row.missing) or 'None'); st.write('Ranking score:',round(row.score,3))
                if 'original_ingredients' in row:
                    st.write('Original ingredient names (quantities unavailable):',json.loads(row.original_ingredients))
                    for step in json.loads(row.directions): st.write(step)
                    st.write('Source:',row.source_link)
                if row.missing and st.button('Add missing to grocery list',key=f'add{i}'):
                    for x in row.missing: db('INSERT OR IGNORE INTO grocery VALUES (?)',(x,))
    st.subheader('Suggested staples')
    st.dataframe(pd.DataFrame(e.staples(inv),columns=['Ingredient','Newly unlocked recipes (sequential)','Dataset frequency']),hide_index=True)
    with st.expander('Ingredient association rules'):
        st.dataframe(pd.DataFrame(e.associations(inv)[:15],columns=['Given','Suggested','Support','Confidence','Lift']),hide_index=True)
st.subheader('Grocery list (local, shared by users of this installation)')
with st.form('grocery'):
    item=st.text_input('Add item')
    if st.form_submit_button('Add') and item.strip(): db('INSERT OR IGNORE INTO grocery VALUES (?)',(item.strip().lower(),))
for (item,) in db('SELECT item FROM grocery ORDER BY item'):
    if st.checkbox(item,key='grocery_'+item):
        db('DELETE FROM grocery WHERE item=?',(item,)); st.rerun()
