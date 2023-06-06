from ropgen_transform import srcml_interface
from ropgen_transform.code_transformers import (
    VarNameStyleTransformer,
    LoopTransformer,
    IncrementTransformer,
    LocalVarDefTransformer,
    LocalVarInitTransformer,
    MultiDefinitionTransformer,
    IncrAssignTransformer,
)

if __name__ == '__main__':
    transformer = LocalVarDefTransformer()

    INPUT_FILE_PATH = './data/test_c/AbstractCodepageDetector.java'
    INPUT_XML_PATH = './data/test_c/AbstractCodepageDetector.xml'
    OUTPUT_XML_PATH = './data/test_c/AbstractCodepageDetector_t.xml'
    OUTPUT_FILE_PATH = './data/test_c/AbstractCodepageDetector_t.java'

    srcml_interface.srcml_program_to_xml(INPUT_FILE_PATH, INPUT_XML_PATH)
    transformer.xml_transform_all(INPUT_XML_PATH, '6.2', OUTPUT_XML_PATH)
    srcml_interface.srcml_xml_to_program(OUTPUT_XML_PATH, OUTPUT_FILE_PATH)
