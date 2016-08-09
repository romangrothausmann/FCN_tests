## FCN inference testing base on https://github.com/developmentseed/caffe-fcn/blob/master/src/fcn-fwd.ipynb


all : caffe-fcn/images/cat.jpg.png

fcn-8s-pascalcontext.caffemodel : %.caffemodel :
	http_proxy="http://proxy.mh-hannover.de:8080" \
	wget \
		-U 'Mozilla/5.0 (X11; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0' \
		http://dl.caffe.berkeleyvision.org/$@ # problems with sophos, use firefox http://dl.caffe.berkeleyvision.org/
#		http://dl.caffe.berkeleyvision.org/pascalcontext-fcn8s-heavy.caffemodel

classify.py : caffe-fcn/fcn-8s/fcn-8s-pascalcontext.caffemodel

%.png : % classify.py
	CAFFE_ROOT=/opt/compilation/caffe/ \
	CAFFE_CPU_MODE=1 \
	/usr/bin/time -v \
	python classify.py $< $@

