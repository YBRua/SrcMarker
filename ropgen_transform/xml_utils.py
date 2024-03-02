from lxml import etree

XML_NS = {
    "src": "http://www.srcML.org/srcML/src",
    "cpp": "http://www.srcML.org/srcML/cpp",
    "pos": "http://www.srcML.org/srcML/position",
}


def load_doc(file_path: str):
    doc = etree.parse(file_path)
    return doc


def init_parser(doc):
    evaluator = etree.XPathEvaluator(doc)
    for ns, val in XML_NS.items():
        evaluator.register_namespace(ns, val)
    return evaluator
