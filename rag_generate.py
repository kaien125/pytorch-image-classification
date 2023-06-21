from skimage import data, segmentation, io
from skimage.future import graph
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray



img = Image.open('pytorch-image-classification/image_ppt/g_min_l_plus.png')
new_size = (112, 112)

img = img.resize(new_size)
# asarray() class is used to convert
# PIL images into NumPy arrays
img = asarray(img)[:,:,:3]
# A colormap to draw the edge
cmap = plt.get_cmap('autumn')

# Labelled image
qs_labels = segmentation.quickshift(img, ratio=0.5)

# Image with boundaries marked
qs_marked = segmentation.mark_boundaries(img, qs_labels,color=(0,0,0))

# Constructing the RAG
qs_graph = graph.rag_mean_color(img, qs_labels)

#Drawing the RAG
qs_out = graph.show_rag(qs_labels, qs_graph , qs_marked)

# plt.imshow(qs_out)
plt.show()



# #Load Image
# img = data.coffee()

# #Segment image
# labels = segmentation.slic(img, compactness=30, n_segments=800)
# #Create RAG
# g = graph.rag_mean_color(img, labels)
# gplt = graph.show_rag(labels, g, img)

# #Draw RAG
# labels = segmentation.quickshift(img, kernel_size=5, max_dist=5, ratio=0.5)
# g = graph.rag_mean_color(img, labels)
# gplt = graph.show_rag(labels, g, img)

# labels = segmentation.felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
# g = graph.rag_mean_color(img, labels)
# gplt = graph.show_rag(labels, g, img)


# io.imshow(gplt)
# io.show()



