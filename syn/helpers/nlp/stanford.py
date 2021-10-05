import os

import jnius_config

from app.config.config import DevelopmentConfig

stanford_class_path = os.environ.get('STANFORD_CORENLP_PATH', DevelopmentConfig.STANFORD_CORENLP_PATH)

jnius_config.set_classpath('.', str(stanford_class_path) + '/*')

from jnius import autoclass

System = autoclass('java.lang.System')
String = autoclass('java.lang.String')
List = autoclass('java.util.List')
ArrayList = autoclass('java.util.ArrayList')
HashMap = autoclass('java.util.HashMap')
Properties = autoclass('java.util.Properties')

CoreAnnotations = autoclass('edu.stanford.nlp.ling.CoreAnnotations')
CoreLabel = autoclass('edu.stanford.nlp.ling.CoreLabel')
Annotation = autoclass('edu.stanford.nlp.pipeline.Annotation')
StanfordCoreNLP = autoclass('edu.stanford.nlp.pipeline.StanfordCoreNLP')
CollapseUnaryTransformer = autoclass('edu.stanford.nlp.sentiment.CollapseUnaryTransformer')
Tree = autoclass('edu.stanford.nlp.trees.Tree')
TreeCoreAnnotations = autoclass('edu.stanford.nlp.trees.TreeCoreAnnotations')
Trees = autoclass('edu.stanford.nlp.trees.Trees')
CoreMap = autoclass('edu.stanford.nlp.util.CoreMap')
StringUtils = autoclass('edu.stanford.nlp.util.StringUtils')
SentencesAnnotation = autoclass('edu.stanford.nlp.ling.CoreAnnotations$SentencesAnnotation')
TokensAnnotation = autoclass('edu.stanford.nlp.ling.CoreAnnotations$TokensAnnotation')
TreeAnnotation = autoclass('edu.stanford.nlp.trees.TreeCoreAnnotations$TreeAnnotation')
BinarizedTreeAnnotation = autoclass('edu.stanford.nlp.trees.TreeCoreAnnotations$BinarizedTreeAnnotation')


def get_collapsed_unary_binary_trees(text):
    tokensProperties = Properties()
    tokensProperties.put("annotators", "tokenize, ssplit")
    tokensPipeline = StanfordCoreNLP(tokensProperties)

    sentencesProperties = Properties()
    sentencesProperties.put("annotators", "tokenize")
    sentencesPipeline = StanfordCoreNLP(sentencesProperties)

    properties = Properties()
    properties.put("annotators", "tokenize, ssplit, pos, lemma, parse")
    properties.setProperty("parse.model", "edu/stanford/nlp/models/srparser/englishSR.ser.gz")
    properties.setProperty("parse.maxlen", "150")
    properties.setProperty("parse.binaryTrees", "True")

    pipeline = StanfordCoreNLP(properties)

    transformer = CollapseUnaryTransformer()

    document = Annotation(text)
    tokensPipeline.annotate(document)

    sentences = document.get(SentencesAnnotation)

    rejected = False
    tokens = ArrayList()
    maxNumTokens = 150
    trees = []
    for sent in sentences:
        sentAnnotation = Annotation(sent.toString())
        sentencesPipeline.annotate(sentAnnotation)
        tokens.add(sentAnnotation.get(TokensAnnotation).size())

        if sentAnnotation.get(TokensAnnotation).size() > maxNumTokens:
            rejected = True

        if not rejected:
            sentenceAnnotation = Annotation(sent.toString())
            pipeline.annotate(sentenceAnnotation)
            sentencesFromSentenceAnnotation = sentenceAnnotation.get(SentencesAnnotation)
            sentenceFromSentenceAnnotation = sentencesFromSentenceAnnotation.get(0)
            binaryTree = sentenceFromSentenceAnnotation.get(BinarizedTreeAnnotation)
            collapsedUnary = transformer.transformTree(binaryTree)
            Trees.convertToCoreLabels(collapsedUnary)
            collapsedUnary.indexSpans()

            trees.append(collapsedUnary.toString())

    return trees
