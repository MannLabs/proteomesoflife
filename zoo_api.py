from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import functions_to_web as zoo

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('num_species')
parser.add_argument('query_species')
parser.add_argument('query_proteins')
parser.add_argument('query_protein')
parser.add_argument('query_type')
parser.add_argument('query_functions')
parser.add_argument('score_cutoff')

class HomologyGraph(Resource):
    def get(self, num_species):
        graph = zoo.generate_graph(num_species=num_species, visualize='web')
        
        return graph


    def post(self):
        num_species = 25
        args = parser.parse_args()
        if 'num_species' in args:
            num_species = args['num_species']
        graph = zoo.generate_graph(num_species=num_species, visualize='web')
        
        return graph, 201


class HomologyGraph2(Resource):
    def get(self, query_protein, query_species, query_type):
        graph = zoo.get_homologs(query_protein=query_protein, query_species=query_species, query_type=query_type, visualize='web')
        
        return graph


    def post(self):
        query_protein='O14983'
        query_species='29760, 9615'
        query_type=['All','Orthologs','Paralogs']
        visualize='web'
        args = parser.parse_args()
        query_protein = args['query_protein']
        query_species = args['query_species']
        query_type = args['query_type']
        graph = zoo.get_homologs(query_protein=query_protein, query_species=query_species, query_type=query_type, visualize='web')
        
        return graph, 201



class RankSpecies(Resource):
    def get(self, query_proteins, query_species):
        rank_plot = zoo.get_species_protein_rank(query_species=query_species, visualize='web', query_proteins=query_proteins)
        
        return rank_plot

    def post(self):
        query_species='9606, 10090'
        query_proteins='Smad7,Ahnak,Ctsa,MYH1,GOT2'
        args = parser.parse_args()
        query_species = args['query_species']
        query_proteins = args['query_proteins']
        
        rank_plot = zoo.get_species_protein_rank(query_species=query_species, visualize='web', query_proteins=query_proteins)

        return rank_plot, 201


class RankFunctions(Resource):
    def get(self, query_species, query_functions):
        function_plot = zoo.get_species_protein_rank_with_function(query_species=query_species, visualize='web', query_functions=query_functions)
        
        return function_plot

    def post(self):
        query_species='9606, 10090'
        query_functions='protein folding'
        args = parser.parse_args()
        query_species = args['query_species']
        query_functions = args['query_functions']
        
        function_plot = zoo.get_species_protein_rank_with_function(query_species=query_species, visualize='web', query_functions=query_functions)

        return function_plot, 201

class HomologsBarplot(Resource):
    def get(self, query_species, query_protein):
        homologs_bar_plot = zoo.plot_homologs_barplot(query_protein=query_protein, query_species=query_species)
        
        return homologs_bar_plot

    def post(self):
        query_species='10090,10116'
        query_protein='O14983'
        args = parser.parse_args()
        query_species = args['query_species']
        query_protein = args['query_protein']
        
        homologs_bar_plot = zoo.plot_homologs_barplot(query_protein=query_protein, query_species=query_species)

        return homologs_bar_plot, 201

class PredictProteinFunction(Resource):
    def get(self, query_protein, score_cutoff):
        predict_funct = zoo.predict_protein_function(query_protein=query_protein, score_cutoff=float(score_cutoff), visualize='web')
        
        return predict_funct

    def post(self):
        query_protein='O14983'
        score_cutoff=0.1
        args = parser.parse_args()
        query_protein = args['query_protein']
        score_cutoff = float(args['score_cutoff'])
        
        predict_funct = zoo.predict_protein_function(query_protein=query_protein, score_cutoff=score_cutoff, visualize='web')

        return predict_funct, 201

class FunctionalGroups(Resource):
    def get(self, query_species, query_functions):
        functional_groups = zoo.get_functional_groups_overview(query_species=query_species, query_functions=query_functions, visualize='polar')
        
        return functional_groups

    def post(self):
        query_species='9606, 10090, 10116, 224325, 572546, 589924, 272569, 1435377, 273057, 9031, 3847, 1620505, 55529'
        query_functions='carbohydrate metabolic process, glycolytic process, ion transport, oxidation-reduction process,photosynthesis,protein folding,proteolysis,translation,translational elongation'
        args = parser.parse_args()
        query_species = args['query_species']
        query_functions = args['query_functions']
        
        functional_groups = zoo.get_functional_groups_overview(query_species=query_species, query_functions=query_functions, visualize='polar')

        return functional_groups, 201

api.add_resource(HomologyGraph, '/homology/<num_species>')
api.add_resource(HomologyGraph2, '/homology2/<query_protein>/<query_species>/<query_type>')
api.add_resource(RankSpecies, '/rank/<query_species>/<query_proteins>')
api.add_resource(RankFunctions, '/rankfunctions/<query_species>/<query_functions>')
api.add_resource(HomologsBarplot, '/homologsbar/<query_protein>/<query_species>')
api.add_resource(PredictProteinFunction, '/predictfunction/<query_protein>/<score_cutoff>')
api.add_resource(FunctionalGroups, '/functionalgroups/<query_species>/<query_functions>')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
