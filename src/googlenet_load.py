import os

def concat_feature(saved_tensor_path):
    tensor_set = [os.path.join(saved_tensor_path, file) for file in os.listdir(saved_tensor_path) ]
    print(sorted(tensor_set[:5]))
    return []

if __name__ == '__main__':
    final_feature = concat_feature('saved_tensor')
    # print(final_feature.shape)