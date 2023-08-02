class RopGenCodeTransformer:
    def get_available_transforms(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    def xml_transform(self, input_xml_path: str, src_style: str, dst_style: str,
                      output_xml_path: str):
        raise NotImplementedError("This method should be implemented by subclasses")

    def xml_transform_all(self, input_xml_path: str, dst_style: str,
                          output_xml_path: str):
        raise NotImplementedError("This method should be implemented by subclasses")

    def etree_transform_all(self, evaluator, dst_style: str):
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_program_style(self, input_xml_path: str):
        raise NotImplementedError("This method should be implemented by subclasses")
