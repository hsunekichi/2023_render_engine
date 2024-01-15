import sys

def count_triangles_in_obj(obj_file_path):
    try:
        with open(obj_file_path, 'r') as obj_file:
            triangle_count = 0
            for line in obj_file:
                if line.startswith('f'):
                    vertices = line.strip().split()[1:]  # Exclude the 'f' and split by whitespace
                    if len(vertices) == 3:
                        triangle_count += 1
                    elif len(vertices) > 3:
                        # If more than 3 vertices are specified in a face, split into triangles
                        triangle_count += len(vertices) - 2
            return triangle_count
    except FileNotFoundError:
        print(f"File '{obj_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    obj_file_path = sys.argv[1]+".obj"  # Replace with the path to your OBJ file
    triangle_count = count_triangles_in_obj(obj_file_path)
    if triangle_count is not None:
        print(f"Number of triangles in the OBJ file: {triangle_count}")
