## FCN inference testing base on https://github.com/developmentseed/caffe-fcn/blob/master/src/fcn-fwd.ipynb


all : caffe-fcn/images/cat.jpg.png mxnet-fcn/images/cat.jpg.png

caffe-fcn/fcn-8s/fcn-8s-pascalcontext.caffemodel : %.caffemodel :
	http_proxy="http://proxy.mh-hannover.de:8080" \
	wget \
		-U 'Mozilla/5.0 (X11; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0' \
		http://dl.caffe.berkeleyvision.org/$@ # problems with sophos, use firefox http://dl.caffe.berkeleyvision.org/
#		http://dl.caffe.berkeleyvision.org/pascalcontext-fcn8s-heavy.caffemodel

caffe-fcn/fcn-8s/legend.txt :
	wget http://www.cs.stanford.edu/~roozbeh/pascal-context/59_labels.txt -O $@

caffe-fcn/classify.py : caffe-fcn/fcn-8s/fcn-8s-pascalcontext.caffemodel

caffe-fcn/images/cat.jpg.png : %.png : % caffe-fcn/classify.py caffe-fcn/fcn-8s/legend.txt
	CAFFE_ROOT=/opt/compilation/caffe/ \
	CAFFE_CPU_MODE=1 \
	/usr/bin/time -v \
	python caffe-fcn/classify.py $< $@

mxnet-fcn/FCN8s_VGG16-symbol.json :
	http_proxy="http://proxy.mh-hannover.de:8080" \
	wget -P $(dir $@) -c \
		'https://www.dropbox.com/sh/578n5cxej7ofd6m/AAA9SFCBN8R_uL2CnAd3WQ5ia/FCN8s_VGG16-symbol.json'

mxnet-fcn/FCN8s_VGG16-0019.params :
	http_proxy="http://proxy.mh-hannover.de:8080" \
	wget -O $@ -c \
		'https://www.dropbox.com/sh/578n5cxej7ofd6m/AABHWZHCtA2P6iR6LUflkxb_a/FCN8s_VGG16-0019-cpu.params' # dropbox has only *-cpu.params, which works with FCN8s_VGG16-symbol.json when saved without "-cpu" -> -O

mxnet-fcn/image_segmentaion.py : mxnet-fcn/FCN8s_VGG16-symbol.json mxnet-fcn/FCN8s_VGG16-0019.params caffe-fcn/fcn-8s/legend.txt

mxnet-fcn/images/cat.jpg.png : %.png : % mxnet-fcn/image_segmentaion.py
	PYTHONPATH=/opt/compilation/mxnet/python/:$$PYTHONPATH \
	/usr/bin/time -v \
	python mxnet-fcn/image_segmentaion.py $< $@
