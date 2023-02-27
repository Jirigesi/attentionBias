from tree_sitter import Language, Parser


Language.build_library(
	# Store the library in the `build` directory
	'build/my-languages.so',
	
	# Include one or more languages
	[
		'/Users/qihongchen/Desktop/tree-sitter-java'
	]
)

JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
parser = Parser()
parser.set_language(JAVA_LANGUAGE)

tree = parser.parse(bytes("""
class Simple{  
	public static void main(String args[]){  
	System.out.println("Hello Java");  
	}  
}  
""", "utf8"))

root_node = tree.root_node

print("sexp = ", root_node.sexp())
function_node = root_node.children[0]
for node in function_node.children:
	print(node)