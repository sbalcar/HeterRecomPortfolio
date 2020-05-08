#!/usr/bin/python3

from configuration.arguments import Arguments #class
from configuration.argument import Argument #class

from recommender.description.recommenderDescription import RecommenderDescription #class

from recommendation.resultOfRecommendation import ResultOfRecommendation #class

from recommender.dummy.recommenderDummyRedirector import RecommenderDummyRedirector #class
from recommender.dummy.recommenderDummyRandom import RecommenderDummyRandom #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from aggregation.aggrDHont import AggrDHont #class
from aggregation.aggrBanditTS import AggrBanditTS #class


from evaluationOfRecommender.evaluationOfRecommenders import EvaluationOfRecommenders #class


class Input:
    def input01():

        resultOfMeth1:ResultOfRecommendation = ResultOfRecommendation([32,2,8,1,4], [0.2,0.1,0.3,0.3,0.1])
        resultOfMeth2:ResultOfRecommendation = ResultOfRecommendation([1,5,32,6,7], [0.1,0.1,0.2,0.3,0.3])
        resultOfMeth3:ResultOfRecommendation = ResultOfRecommendation([7,2,77,64,12], [0.3,0.1,0.2,0.3,0.1])

        recomm1:RecommenderDescription = RecommenderDescription(RecommenderDummyRedirector, Arguments([Argument("RESULT", resultOfMeth1)]));
        recomm2:RecommenderDescription = RecommenderDescription(RecommenderDummyRedirector, Arguments([Argument("RESULT", resultOfMeth2)]));
        recomm3:RecommenderDescription = RecommenderDescription(RecommenderDummyRedirector, Arguments([Argument("RESULT", resultOfMeth3)]));

        aggr:AggregationDescription = AggregationDescription(AggrDHont, Arguments([]));

        portfolioDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(["metoda1", "metoda2", "metoda3"], [recomm1, recomm2, recomm3], aggr)

        evaluationOfRecommenders:EvaluationOfRecommenders = EvaluationOfRecommenders()
        evaluationOfRecommenders.addArg('metoda1', Argument("votes", 100))
        evaluationOfRecommenders.addArg('metoda2', Argument("votes", 80))
        evaluationOfRecommenders.addArg('metoda3', Argument("votes", 60))


        # (PortfolioDescription, EvaluationOfRecommenders)
        return (portfolioDescr, evaluationOfRecommenders);


    def input02():

        # resultOfMeth1:ResultOfRecommendation
        resultOfMeth1 = ResultOfRecommendation([32,2,8,1,4], [0.9,0.01,0.03,0.03,0.01])
        resultOfMeth2 = ResultOfRecommendation([1,5,32,6,7], [0.9,0.01,0.6,0.03,0.03])
        resultOfMeth3 = ResultOfRecommendation([7,2,77,64,12], [0.9,0.01,0.02,0.03,0.01])

        # recomm1:RecommenderDescription
        recomm1 = RecommenderDescription(RecommenderDummyRedirector, Arguments([Argument("RESULT", resultOfMeth1)]));
        recomm2 = RecommenderDescription(RecommenderDummyRedirector, Arguments([Argument("RESULT", resultOfMeth2)]));
        recomm3 = RecommenderDescription(RecommenderDummyRedirector, Arguments([Argument("RESULT", resultOfMeth3)]));

        aggr = AggregationDescription(AggrBanditTS, Arguments([]));

        # portfolioDescr:PortfolioDescription
        portfolioDescr = Portfolio1AggrDescription(["metoda1", "metoda2", "metoda3"], [recomm1, recomm2, recomm3], aggr)

        # a1:list<Argument>
        a1 = [Argument("r", 5), Argument("n", 10), Argument("alpha0", 1), Argument("beta0", 1)]
        a2 = [Argument("r", 5), Argument("n", 10), Argument("alpha0", 1), Argument("beta0", 1)]
        a3 = [Argument("r", 6), Argument("n", 130), Argument("alpha0", 1), Argument("beta0", 1)]

        # evaluationOfRecommenders:EvaluationOfRecommenders
        evaluationOfRecommenders = EvaluationOfRecommenders()
        evaluationOfRecommenders.add('metoda1', Arguments(a1))
        evaluationOfRecommenders.add('metoda2', Arguments(a2))
        evaluationOfRecommenders.add('metoda3', Arguments(a3))


        # (PortfolioDescription, EvaluationOfRecommenders)
        return (portfolioDescr, evaluationOfRecommenders);


    def input02_():

        RecommenderDummyRedirector

        recomm1 = RecommenderDescription(RecommenderDummyRandom, Arguments([]));

        aggr = AggregationDescription(AggrDHont, Arguments([]));

        portfolioDesr = Portfolio1AggrDescription(["metoda1", "metoda2", "metoda3"], [recomm1, recomm1, recomm1], aggr)

        # portfolioDesr:PortfolioDescription
        return portfolioDesr;
