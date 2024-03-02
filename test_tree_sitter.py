import os
from metrics.syntax_match import pprint_tree, check_tree_validity
from tree_sitter import Parser, Language

if __name__ == "__main__":
    root_dir = "./metrics"
    lang = "cpp"
    dataset_dir = "./data/github_c"

    MAX_DEPTH = -1
    JAVA_LANGUAGE = Language(root_dir + "/parser/languages.so", lang)
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)

    code = """int main ( ) { freopen ( " A-small-attempt0.in " , " r " , stdin ) ; freopen ( " A-small-attempt0.out " , " w " , stdout ) ; int T ; scanf ( " %d " , & T ) ; for ( int test = 1 ; test <= T ; ++ test ) { scanf ( " %lf%d " , & L , & n ) ; double t = 0 ; for ( int i = 0 ; i < n ; ++ i ) { scanf ( " %lf%lf " , & a [ i ] , & v [ i ] ) ; t = max ( t , ( L - a [ i ] ) / v [ i ] ) ; } double Case = L / t ; printf ( " Case #%d: %.12lf\n " , test , Case ) ; } return 0 ; }"""

    tree = parser.parse(bytes(code, "utf-8"))
    pprint_tree(tree.root_node)
    print(check_tree_validity(tree.root_node, max_depth=MAX_DEPTH))
    print(tree.root_node.has_error)

    # tot_files = 0
    # good_files = 0
    # for author in os.listdir(dataset_dir):
    #     for file in os.listdir(os.path.join(dataset_dir, author)):
    #         with open(os.path.join(dataset_dir, author, file), 'r') as f:
    #             code = f.read()
    #             tree = parser.parse(bytes(code, 'utf-8'))
    #             is_valid = check_tree_validity(tree.root_node, max_depth=MAX_DEPTH)

    #             if is_valid:
    #                 good_files += 1
    #             else:
    #                 print(os.path.join(dataset_dir, author, file))
    #                 # pprint_tree(tree.root_node)
    #             tot_files += 1

    # print(f'Valid files: {good_files}/{tot_files} ({good_files/tot_files*100:.2f}%)')
