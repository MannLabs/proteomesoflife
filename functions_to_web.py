#!/usr/bin/env python
# coding: utf-8


from py2neo import Graph
import pandas as pd
import os
import re
import networkx as nx
from networkx.readwrite import json_graph
import ast
import math
from plotly import subplots
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot
from cyjupyter import Cytoscape

##Database parameters
host = ""
port = 7687
user = ""
password = ""

driver = Graph(host=host, port=port, user=user, password=password)


# ## Network functions

def visualize_cys(path):
    pathcys, layout, stylesheet = networkx_to_cytoscape(path)
    pathvis = Cytoscape(data={'elements':pathcys}, visual_style=stylesheet, layout=layout)

    return pathvis

def get_cys_graph(path):
    elements, layout, stylesheet = networkx_to_cytoscape(path)

    return elements, layout, stylesheet

def get_nx_graph(query_path):
    G = None
    regex = r"\(?(.+)\)\<?\-\>?\[\:(.+)\s\{.*\}\]\<?\-\>?\((.+)\)?"
    nodes = set()
    rels = set()
    for r in query_path:
        path = str(r['path'])
        matches = re.search(regex, path)
        if matches:
            source = matches.group(1)
            source_match = re.search(regex, source)
            if source_match:
                source = source_match.group(1)
                relationship = source_match.group(2)
                target = source_match.group(3)
                nodes.update([source, target])
                rels.add((source, target, relationship))
                source = target
            relationship = matches.group(2)
            target = matches.group(3)
            nodes.update([source, target])
            rels.add((source, target, relationship))
    if len(nodes)> 0:
        G = nx.Graph()
        G.add_nodes_from(nodes, color='blue')
        for s,t,label in rels:
            G.add_edge(s,t,label=label, width=1)

    return G

def networkx_to_cytoscape(graph):
    cy_graph = json_graph.cytoscape_data(graph)
    cy_nodes = cy_graph['elements']['nodes']
    cy_edges = cy_graph['elements']['edges']
    cy_elements = cy_nodes
    cy_elements.extend(cy_edges)
    layout = {'name': 'cose',
                'idealEdgeLength': 100,
                'nodeOverlap': '4',
                'refresh': 20,
                'fit': True,
                #'padding': 30,
                'randomize': True,
                'componentSpacing': 100,
                'nodeRepulsion': 400000,
                'edgeElasticity': 100,
                'nestingFactor': 5,
                'gravity': '10',
                'numIter': 1000,
                'initialTemp': 200,
                'coolingFactor': 0.95,
                'minTemp': 1.0,
                 'height':'1000','width':'1400'}

    stylesheet = [{'selector': 'node', 'style': {'label': 'data(name)', 'width':5, 'height':5,'font-size':1, 'background-color':'#3182bd'}}, 
                {'selector':'edge','style':{'curve-style': 'bezier', 'width':0.5}}]

    return cy_elements, layout, stylesheet

def give_cytonet_style(node_colors):
    color_selector = "{'selector': '[name = \"KEY\"]', 'style': {'background-color': 'VALUE'}}"
    stylesheet=[{'selector': 'node', 'style': {'label': 'data(name)','width':5, 'height':5,'font-size':1,}}, 
                {'selector':'edge','style':{'curve-style': 'bezier', 'width':0.5}}]
    
    stylesheet.extend([{'selector':'[width < 1.1]', 'style':{'line-color':'#ef3b2c', 'line-style': 'dashed', 'width':'width'}},{'selector':'[width = 1.1]', 'style':{'line-color':'#2171b5','width':'width'}},{'selector':'[width > 1.1]', 'style':{'line-color':'#ae017e','width':'width'}}])

    for k,v in node_colors.items():
        stylesheet.append(ast.literal_eval(color_selector.replace("KEY", k.replace("'","\'")).replace("VALUE",v)))
    
    return stylesheet

# ## Plotting functions

def get_simple_scatterplot(data, args):
    figure = {}
    m = {'size': 15, 'line': {'width': 0}}
    text = data.name
    if 'colors' in data.columns:
        m.update({'color':data['colors'].tolist()})
    if 'size' in data.columns:
        m.update({'size':data['size'].tolist()})
    if 'symbol' in data.columns:
        m.update({'symbol':data['symbol'].tolist()})
    
    annots=[]
    if 'annotations' in args:
        for index, row in data.iterrows():
            name = row['name'].split(' ')[0]
            if name in args['annotations']:
                annots.append({'x': row['x'], 
                            'y': row['y'], 
                            'xref':'x', 
                            'yref': 'y', 
                            'text': name, 
                            'showarrow': False, 
                            'ax': 55, 
                            'ay': -1,
                            'font': dict(size = 8)})
    figure['data'] = [go.Scattergl(x = data.x,
                                y = data.y,
                                text = text,
                                mode = 'markers',
                                opacity=0.7,
                                marker= m,
                                )]
                                
    figure["layout"] = go.Layout(title = args['title'],
                                xaxis= {"title": args['x_title']},
                                yaxis= {"title": args['y_title']},
                                margin={'l': 40, 'b': 40, 't': 30, 'r': 10},
                                legend={'x': 0, 'y': 1},
                                hovermode='closest',
                                height=args['height'],
                                width=args['width'],
                                annotations = annots + [dict(xref='paper', yref='paper', showarrow=False, text='')],
                                showlegend=False,
                                template='plotly_white'
                                )
    
    return figure

def generate_rank(data, args):
    num_cols = 2
    fig = {}
    layouts = []
    num_groups = len(data.index.unique())
    num_rows = math.ceil(num_groups/num_cols)
    fig = subplots.make_subplots(rows=num_rows, cols=num_cols, shared_yaxes=True,print_grid=False)
    r = 1
    c = 1
    range_y = [data['Value'].min(), data['Value'].max()+1]
    for index in data.index.unique():
        gdata = data.loc[index, :].dropna(how='all').groupby('Name', as_index=False).mean().sort_values(by='Value', ascending=False)
        gdata = gdata.reset_index().reset_index()
        cols = ['x', 'group', 'name', 'y']
        cols.extend(gdata.columns[4:])
        gdata.columns = cols
        gfig = get_simple_scatterplot(gdata, args)
        trace = gfig['data'].pop()
        glayout = gfig['layout']['annotations']

        for l in glayout:
            nlayout = dict(x = l.x,
                        y = l.y,
                        xref = 'x'+str(c),
                        yref = 'y'+str(r),
                        text = l.text,
                        showarrow = True,
                        ax = l.ax,
                        ay = l.ay,
                        font = l.font,
                        align='center',
                        arrowhead=1,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor='#636363')
            layouts.append(nlayout)
        trace.name = index
        fig.append_trace(trace, r, c)

        if c >= num_cols:
            r += 1
            c = 1
        else:
            c += 1
    fig['layout'].update(dict(height = args['height'], 
                            width=args['width'],  
                            title=args['title'], 
                            xaxis= {"title": args['x_title'], 'autorange':True}, 
                            yaxis= {"title": args['y_title'], 'range':range_y},
                            template='plotly_white'))
    fig['layout'].annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')] + layouts 

    return fig

def get_barplot(data, args):
    figure = {}
    figure["data"] = []
    if "group" in args:
        for g in data[args["group"]].unique():
            color = None
            if 'colors' in args:
                if g in args['colors']:
                    color = args['colors'][g]
            trace = go.Bar(
                        x = data.loc[data[args["group"]] == g,args['x']], # assign x as the dataframe column 'x'
                        y = data.loc[data[args["group"]] == g, args['y']],
                        name = g,
                        marker = dict(color=color)
                        )
            figure["data"].append(trace)
    else:
        figure["data"].append(
                      go.Bar(
                            x=data[args['x']], # assign x as the dataframe column 'x'
                            y=data[args['y']]
                        )
                    )
    figure["layout"] = go.Layout(
                            title = args['title'],
                            xaxis={"title":args["x_title"]},
                            yaxis={"title":args["y_title"]},
                            height = args['height'],
                            width = args['width'],
                            annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')],
                            template='plotly_white'
                        )

    return figure


# ## Network example

def generate_graph(num_species=25, visualize='notebook'):
    query = '''MATCH path=(p:Project)-[:STUDIES_SPECIE]-(s:Taxonomy{name:s.name})RETURN path LIMIT ''' + str(num_species)
    query_path = driver.run(query).data()
    
    G = get_nx_graph(query_path)
    
    if visualize == 'notebook':
        return visualize_cys(G)
    elif visualize == 'web':
        return get_cys_graph(G)
    

    return None


def get_homologs(query_protein='O14983', query_species='29760, 9615', query_type=['All','Orthologs','Paralogs'], visualize='web'):
    query= r'MATCH path=(p1:Protein)-[r:RELATIONSHIP]-(p2:Protein)-[:BELONGS_TO_TAXONOMY]->(t:Taxonomy) WHERE (p1.id="{}" OR p1.name="{}") AND p2.id<>p1.id'.format(query_protein, query_protein)
    query_species = query_species.split(',')
    query_protein_specie = r'MATCH (p:Protein)-[:BELONGS_TO_TAXONOMY]->(t:Taxonomy) WHERE (p.id="{}" OR p.name="{}") RETURN t.id AS taxid'.format(query_protein, query_protein)
    
    sp_request = driver.run(query_protein_specie).data()
    print("Got response")
    spdf = pd.DataFrame(sp_request)
    if not spdf.empty:
        sp = spdf['taxid'][0]
        query_species.append(sp)
    if '' not in query_species or len(query_species)>1:
        query = query + " AND p2.taxid IN [{}]".format(",".join(['"{}"'.format(s.strip()) for s in query_species]))
    if query_type == "Orthologs":
        query = query.replace("RELATIONSHIP", 'IS_ORTHOLOG')
    elif query_type == 'Paralogs':
        query = query.replace("RELATIONSHIP", 'IS_PARALOG')
    else:
        query = query.replace("RELATIONSHIP", 'IS_ORTHOLOG|IS_PARALOG')
    
    if visualize == "Networkx" or visualize == "web":
        query = query + "RETURN path"
        request = driver.run(query).data()
        path = get_nx_graph(request)
        if path is not None:
            if visualize == "notebook":
                return visualize_nx(path)
            else:
                return get_cys_graph(path)

    return None


def get_species_protein_rank(query_species='9606, 10090', visualize='web', query_proteins='Smad7,Ahnak,Ctsa,MYH1,GOT2'):   
    rank = None
    args={'x_title':'protein ranking', 'y_title':'intensity', 'title':'Ranking of Proteins', 'height':900, 'width':900}
    query= r'MATCH (t:Taxonomy)-[:BELONGS_TO_TAXONOMY]-(pr:Protein)-[r:HAS_QUANTIFIED_PROTEIN]-(p:Project)'
    query_species = query_species.split(',')
    if '' not in query_species or len(query_species)>1:
        query = (query + " WHERE t.id IN [{}] RETURN pr.id AS Identifier, pr.name AS Name, r.value AS Value, t.id AS Taxid").format(",".join(['"{}"'.format(s.strip()) for s in query_species]))
        request = driver.run(query).data()
        table = pd.DataFrame(request)
        if not table.empty:
            table = table.set_index('Taxid').drop_duplicates()
            if visualize == "notebook" or visualize == "web":
                table = table.sort_values('Value',ascending=True)
                query_proteins = [p.strip() for p in query_proteins.split(',')]
                if '' not in query_proteins or len(query_proteins) > 1:
                    args['annotations'] = query_proteins
                rank = generate_rank(table, args)
                if visualize == 'notebook':
                    iplot(rank)
                else:
                    return plot(rank, output_type='div')
            else:
                return table
    return None


def get_species_protein_rank_with_function(query_species='9606, 10090', visualize=['Rank', 'Table'], query_functions='protein folding'):   
    rank = None
    args={'x_title':'protein ranking', 'y_title':'intensity', 'title':'Ranking of Proteins', 'height':1200, 'width':900}
    query= r'MATCH path=(t:Taxonomy)<-[:BELONGS_TO_TAXONOMY]-(pr:Protein)'
    query_species = query_species.split(',')
    if '' not in query_species or len(query_species)>1:
        query = query + " WHERE t.id IN [{}] WITH pr,t MATCH (p:Project)-[r:HAS_QUANTIFIED_PROTEIN]->(pr) RETURN pr.id AS Identifier, pr.name AS Name, r.value AS Value, t.id AS Taxid".format(",".join(['"{}"'.format(s.strip()) for s in query_species]))
        request = driver.run(query).data()
        table = pd.DataFrame(request)
        query_functions = query_functions.split(';')
        if '' not in query_functions or len(query_functions) > 1:
            query_functions = r'MATCH path=(t:Taxonomy)<-[:BELONGS_TO_TAXONOMY]-(pr:Protein)-[:ASSOCIATED_WITH]-(f) WHERE t.id IN [{}] AND toLower(f.name) IN [{}] RETURN DISTINCT(pr.name) AS Name, f.name AS Function'.format(",".join(['"{}"'.format(s.strip()) for s in query_species]),",".join(['"{}"'.format(f.strip()) for f in query_functions]))
            request_functions = driver.run(query_functions).data()
            annotations = pd.DataFrame(request_functions)
            table = table.set_index('Name').join(annotations.set_index('Name')).reset_index()
            args['annotations'] = annotations['Name'].tolist()
        table = table.set_index('Taxid').drop_duplicates()
        if not table.empty:
            if visualize == "Rank":
                table = table.sort_values('Value',ascending=True)
                rank = generate_rank(table, args)
                return plot(rank, output_type='div')
            else:
                return table
        return None

def plot_homologs_barplot(query_protein='O14983', query_species='10090,10116'):
    query= r'MATCH path=(p1:Protein)-[:IS_ORTHOLOG]-(p2:Protein) WHERE (p1.id="{}" OR p1.name="{}") WITH collect(p1.id) AS ids1, collect(p2.id) AS ids2 MATCH (t:Taxonomy)-[:BELONGS_TO_TAXONOMY]-(pr:Protein)-[r:HAS_QUANTIFIED_PROTEIN]-(p:Project) WHERE pr.id IN ids1 OR pr.id IN ids2'.format(query_protein, query_protein)
    query_species = query_species.split(',')
    if '' not in query_species:
        query = query + " AND t.id IN [{}]".format(",".join(['"{}"'.format(s.strip()) for s in query_species]))
    query = query + " RETURN pr.id AS Identifier, r.value AS Value, t.name AS Taxid"
    request = driver.run(query).data()
    table = pd.DataFrame(request)
    if not table.empty:
        title= "Plot Homologs {}".format(query_protein)
        figure = get_barplot(table, args={'group':'Taxid', 'x':'Identifier', 'y':'Value',
                                          'x_title':'Homologs',
                                          'y_title':'Intensity',
                                          'height':600,
                                          'width':1000,
                                          'title':title})
        return plot(figure, output_type='div')

    return None

def predict_protein_function(query_protein='O14983', score_cutoff=0.1, visualize='web'):   
    net = None
    query_total_homologs = r'MATCH path=(p1:Protein)-[:IS_ORTHOLOG]-(p2:Protein) WHERE (p1.id="{}" OR p1.name="{}") AND p2.id<>p1.id WITH p1,p2 MATCH (p2)-[:HAS_QUANTIFIED_PROTEIN]-() RETURN count(DISTINCT(p2.id)) AS total_homologs'.format(query_protein, query_protein)
    query= r'MATCH path=(p1:Protein)-[:IS_ORTHOLOG]-(p2:Protein) WHERE (p1.id="{}" OR p1.name="{}") AND p2.id<>p1.id WITH p1, p2 WITH p1,p2 MATCH (p2)-[:HAS_QUANTIFIED_PROTEIN]-() WITH p1,p2, collect(p2.id) AS ids2 MATCH (p1)-[:ASSOCIATED_WITH]-(f:Biological_process) WITH collect(f.id) AS functions, p1,p2, ids2 MATCH (p2)-[:ASSOCIATED_WITH]-(f:Biological_process) WHERE NOT f.id IN functions RETURN p1.id AS Identifier, p1.name AS Name, f.id AS Function_id, f.name AS Function, count(f.id) AS Num_Orthologs, collect(p2.id) AS Orthologs ORDER BY Num_Orthologs DESC'.format(query_protein, query_protein)
    query_functions= r'MATCH (p1)-[:ASSOCIATED_WITH]-(f:Biological_process) WHERE (p1.id="{}" OR p1.name="{}") RETURN p1.id AS Identifier, p1.name AS Name, f.id AS Function_id, f.name AS Function'.format(query_protein, query_protein)
    
    total_homologs = driver.run(query_total_homologs).data()[0]['total_homologs']
    request = driver.run(query).data()
    table = pd.DataFrame(request)
    if not table.empty:
        table['total_orthologs'] = total_homologs
        table['score'] = table['Num_Orthologs']/total_homologs
        table = table[table['score'] >= float(score_cutoff)]
        if visualize == "Table":
            return table
        elif visualize == "Networkx" or visualize == "Cytoscape" or visualize == 'web':
            nodes = set()
            rels = set()
            functions = set()
            node_colors = {}
            for index, row in table.iterrows():
                source = row['Function'].title()
                functions.add(row['Function_id'])
                target = row['Name']
                node_colors[source] = '#ef3b2c'
                relationship = ('predicted', row['score'], '#ef3b2c')
                nodes.update([source, target])
                rels.add((source, target, relationship))
            request_annotation = driver.run(query_functions).data()
            annotations = pd.DataFrame(request_annotation)
            if not annotations.empty:
                for index, row in annotations.iterrows():
                    source = row['Function'].title()
                    functions.add(row['Function_id'])
                    target = row['Name']
                    node_colors[source] = '#2171b5'
                    relationship = ('annotated', 1.1, '#2171b5')
                    nodes.update([source, target])
                    rels.add((source, target, relationship))
            functions = ",".join(['"{}"'.format(f) for f in functions])
            query_functions_rels = r'MATCH (f1:Biological_process)-[:HAS_PARENT]->(parent:Biological_process)<-[:HAS_PARENT]-(f2:Biological_process) WHERE f1.id IN [{}] AND f2.id IN [{}] AND f1.id<>f2.id RETURN f1.name AS source1, parent.name AS parent, f2.name AS source2'.format(functions, functions)
            request_hierarchy = driver.run(query_functions_rels).data()
            hierarchy = pd.DataFrame(request_hierarchy)
            if not hierarchy.empty:
                for index, row in hierarchy.iterrows():
                    source1 = row['source1'].title()
                    target = row['parent']
                    source2 = row['source2'].title()
                    node_colors[target] = '#ae017e'
                    relationship = ('hierarchy', 2, '#ae017e')
                    nodes.update([target])
                    rels.add((source1, target, relationship))
                    rels.add((source2, target, relationship))

            if len(nodes)> 0:
                G = nx.Graph()        
                G.add_nodes_from(nodes, color='blue')
                for s,t,attrs in rels:
                    G.add_edge(s,t,label=attrs[0], width=attrs[1], color=attrs[2])
                if visualize == "Networkx":
                    return visualize_nx(G)
                else:
                    return get_cys_graph(G)
            
    return table

import plotly.express as px
def get_functional_groups_overview(query_species='9606, 10090, 10116, 224325, 572546, 589924, 272569, 1435377, 273057, 9031, 3847, 1620505, 55529', 
                                   query_functions='carbohydrate metabolic process, glycolytic process, ion transport, oxidation-reduction process,photosynthesis,protein folding,proteolysis,translation,translational elongation',
                                   visualize=['polar', 'table']):
    query = r'MATCH (t:Taxonomy)<-[:BELONGS_TO_TAXONOMY]-(p:Protein) WHERE t.id IN [{}] WITH t, p MATCH (p)<-[r:HAS_QUANTIFIED_PROTEIN]-() WITH t, p, r MATCH (t)-[]-(p)-[:ASSOCIATED_WITH]-(bp:Biological_process) WHERE toLower(bp.name) IN [{}] RETURN t.name AS Taxid, r.value AS Intensity, bp.name AS Function'
    query_species = query_species.split(',')
    query_functions = query_functions.split(',')
    if '' not in query_species or len(query_species)>1:
        query_species = ",".join(['"{}"'.format(s.strip()) for s in query_species])
        if '' not in query_functions or len(query_functions)>1:
            query_functions = ",".join(['"{}"'.format(f.strip()) for f in query_functions])
            query = query.format(query_species, query_functions)
    
    request = driver.run(query).data()
    table = pd.DataFrame(request)
    if not table.empty:
        if visualize == "polar":
            df = table.groupby(['Taxid', 'Function'])
            list_cols = []
            for group in df.groups:
                mean = df.get_group(group).mean()["Intensity"]
                taxid, function = group
                list_cols.append([taxid, function, mean])
            summ_df = pd.DataFrame(list_cols, columns=['Taxid', 'Function', 'avg Intensity'])
            summ_df['Function'] = summ_df['Function'].apply(lambda x: x.title())
            fig = px.line_polar(summ_df, r="avg Intensity", theta="Taxid", color="Function", line_close=True,
                    title='Mean Expression Levels of Functional Groups',
                    color_discrete_sequence=px.colors.sequential.Plasma[-2::-1], width=1000)
            return plot(fig, output_type='div')