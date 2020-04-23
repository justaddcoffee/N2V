import tensorflow as tf

from xn2v.text_encoder_tf import TextEncoder
from xn2v import SkipGramWord2Vec

print(tf.__version__)
assert tf.__version__ >= "2.0"



treasure_island = '/Users/peterrobinson/Downloads/treasure.txt'

encoder = TextEncoder(treasure_island, data_type='sentences')
tensor_data,  count_list, dictionary,  reverse_dictionary = encoder.build_dataset()

#print("count_list size=%d" % len(count_list))
##print("dictionary size=%d" % len(dictionary))
#print("reverse_dictionary size=%d" % len(reverse_dictionary))
# print(tensor_data)

print(type(tensor_data))

batch_size = 128
window_shift=1

sgw2v = SkipGramWord2Vec(data=tensor_data,
                         worddictionary=dictionary,
                         reverse_worddictionary=reverse_dictionary)

sgw2v.train2()

exit(42)

encoder2 = TextEncoder(treasure_island, data_type='words')
tensor_data,  count_list, dictionary,  reverse_dictionary = encoder2.build_dataset()

print("count_list size=%d" % len(count_list))
print("dictionary size=%d" % len(dictionary))
print("reverse_dictionary size=%d" % len(reverse_dictionary))
# print(tensor_data)

print(type(tensor_data))
c = 0
for t in tensor_data:
    print((t.numpy()))
    print(type(t))
    c += 1
    if c > 5:
        break
batch_size = 128
window_shift=1



exit(9)
sgw2v = SkipGramWord2Vec(data=tensor_data,
                         worddictionary=dictionary,
                         reverse_worddictionary=reverse_dictionary)

sgw2v.train(display_step=50)