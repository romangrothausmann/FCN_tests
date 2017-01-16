## FCN inference testing base on https://github.com/developmentseed/caffe-fcn/blob/master/src/fcn-fwd.ipynb


.PHONY: all infer train

all : infer train
infer : caffe-fcn/images/cat.jpg.png mxnet-fcn/images/cat.jpg.png
train : mxnet-fcn/model_pascal/FCN8s_VGG16-symbol.json


.PHONY: %.caffemodel
caffe-fcn/fcn-8s/fcn-8s-pascalcontext.caffemodel : %.caffemodel :
	http_proxy="http://proxy.mh-hannover.de:8080" \
	https_proxy="http://proxy.mh-hannover.de:8080" \
	wget \
		-O $@ -c \
		-U 'Mozilla/5.0 (X11; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0' \
		http://dl.caffe.berkeleyvision.org/$(notdir $@) # problems with sophos, use firefox http://dl.caffe.berkeleyvision.org/
#		http://dl.caffe.berkeleyvision.org/pascalcontext-fcn8s-heavy.caffemodel

.PHONY: caffe-fcn/fcn-8s/legend.txt
caffe-fcn/fcn-8s/legend.txt :
	http_proxy="http://proxy.mh-hannover.de:8080" \
	https_proxy="http://proxy.mh-hannover.de:8080" \
	wget http://www.cs.stanford.edu/~roozbeh/pascal-context/59_labels.txt -O $@

caffe-fcn/classify.py : caffe-fcn/fcn-8s/fcn-8s-pascalcontext.caffemodel

caffe-fcn/images/cat.jpg.png : %.png : % caffe-fcn/classify.py caffe-fcn/fcn-8s/legend.txt
	CAFFE_ROOT=/opt/compilation/caffe/ \
	CAFFE_CPU_MODE=0 \
	/usr/bin/time -v \
	python caffe-fcn/classify.py $< $@

.PHONY: mxnet-fcn/FCN8s_VGG16-symbol.json
mxnet-fcn/FCN8s_VGG16-symbol.json :
	http_proxy="http://proxy.mh-hannover.de:8080" \
	https_proxy="http://proxy.mh-hannover.de:8080" \
	wget -P $(dir $@) -c \
		'https://www.dropbox.com/sh/578n5cxej7ofd6m/AAA9SFCBN8R_uL2CnAd3WQ5ia/FCN8s_VGG16-symbol.json'

.PHONY: mxnet-fcn/FCN8s_VGG16-0019.params
mxnet-fcn/FCN8s_VGG16-0019.params :
	http_proxy="http://proxy.mh-hannover.de:8080" \
	https_proxy="http://proxy.mh-hannover.de:8080" \
	wget -O $@ -c \
		'https://www.dropbox.com/sh/578n5cxej7ofd6m/AABHWZHCtA2P6iR6LUflkxb_a/FCN8s_VGG16-0019-cpu.params' # dropbox has only *-cpu.params, which works with FCN8s_VGG16-symbol.json when saved without "-cpu" -> -O

mxnet-fcn/image_segmentaion.py : mxnet-fcn/FCN8s_VGG16-symbol.json mxnet-fcn/FCN8s_VGG16-0019.params caffe-fcn/fcn-8s/legend.txt

mxnet-fcn/images/cat.jpg.png : %.png : % mxnet-fcn/image_segmentaion.py
	PYTHONPATH=/opt/compilation/mxnet/python/:$$PYTHONPATH \
	/usr/bin/time -v \
	python mxnet-fcn/image_segmentaion.py $< $@



#### training


.PHONY: VOCtrainval_11-May-2012.tar
VOCtrainval_11-May-2012.tar :
	http_proxy="http://proxy.mh-hannover.de:8080" \
	https_proxy="http://proxy.mh-hannover.de:8080" \
	wget -c \
		'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'

VOCdev%/VOC2012/ VOCdev%/VOC2012/JPEGImages/ VOCdev%/VOC2012/SegmentationClass/ VOCdev%/VOC2012/ImageSets/Segmentation/train.txt VOCdev%/VOC2012/ImageSets/Segmentation/val.txt : | VOCtrainval_11-May-2012.tar # | because tar stores acc-times
	tar xvf $< # --strip-components=1

mxnet-fcn/VOC2012 : VOCdevkit/VOC2012/
	ln -s ../$< $@

.PHONY: mxnet-fcn/VGG_FC_ILSVRC_16_layers-symbol.json
mxnet-fcn/VGG_FC_ILSVRC_16_layers-symbol.json :
	http_proxy="http://proxy.mh-hannover.de:8080" \
	https_proxy="http://proxy.mh-hannover.de:8080" \
	wget -O $@ -c \
		'https://www.dropbox.com/sh/578n5cxej7ofd6m/AABS-VGdlyuhfW9T9nXu7aNza/VGG_FC_ILSVRC_16_layers-symbol.json?dl=0'

.PHONY: mxnet-fcn/VGG_FC_ILSVRC_16_layers-0074.params
mxnet-fcn/VGG_FC_ILSVRC_16_layers-0074.params :
	http_proxy="http://proxy.mh-hannover.de:8080" \
	https_proxy="http://proxy.mh-hannover.de:8080" \
	wget -O $@ -c \
		'https://www.dropbox.com/sh/578n5cxej7ofd6m/AABF9AK6y1nf2EnxjsJyxp4Ja/VGG_FC_ILSVRC_16_layers-0074.params?dl=0'

mxnet-fcn/model_pascal/ :
	mkdir -p $@

mxnet-fcn/VOC2012/JPEGImages/ mxnet-fcn/VOC2012/SegmentationClass/ : | mxnet-fcn/VOC2012

mxnet-fcn/VOC2012/%.lst : VOCdevkit/VOC2012/ImageSets/Segmentation/%.txt | mxnet-fcn/VOC2012/JPEGImages/ mxnet-fcn/VOC2012/SegmentationClass/
	awk '{printf("%d\tJPEGImages/%s.jpg\tSegmentationClass/%s.png\n", NR, $$1, $$1)}' $< > $@

mxnet-%/model_pascal/FCN32s_VGG16-0050.params mxnet-%/model_pascal/FCN32s_VGG16-symbol.json : mxnet-fcn/VGG_FC_ILSVRC_16_layers-0074.params mxnet-fcn/VGG_FC_ILSVRC_16_layers-symbol.json   mxnet-fcn/VOC2012/train.lst mxnet-fcn/VOC2012/val.lst  | mxnet-fcn/model_pascal/
	cd mxnet-fcn/ && \
	PYTHONPATH=/opt/compilation/mxnet/python/:$$PYTHONPATH \
	/usr/bin/time -v \
	python -u fcn_xs.py --model=fcn32s --prefix=VGG_FC_ILSVRC_16_layers  --epoch=74 --init-type=vgg16 # from mxnet-fcn/run_fcnxs.sh

mxnet-%/model_pascal/FCN16s_VGG16-0050.params mxnet-%/model_pascal/FCN16s_VGG16-symbol.json : mxnet-fcn/model_pascal/FCN32s_VGG16-0050.params mxnet-fcn/model_pascal/FCN32s_VGG16-symbol.json   mxnet-fcn/VOC2012/train.lst mxnet-fcn/VOC2012/val.lst  | mxnet-fcn/model_pascal/
	cd mxnet-fcn/ && \
	PYTHONPATH=/opt/compilation/mxnet/python/:$$PYTHONPATH \
	/usr/bin/time -v \
	python -u fcn_xs.py --model=fcn16s --prefix=model_pascal/FCN32s_VGG16 --epoch=31 --init-type=fcnxs # from mxnet-fcn/run_fcnxs.sh

mxnet-%/model_pascal/FCN8s_VGG16-0050.params mxnet-%/model_pascal/FCN8s_VGG16-symbol.json : mxnet-fcn/model_pascal/FCN16s_VGG16-0050.params mxnet-fcn/model_pascal/FCN16s_VGG16-symbol.json   mxnet-fcn/VOC2012/train.lst mxnet-fcn/VOC2012/val.lst  | mxnet-fcn/model_pascal/
	cd mxnet-fcn/ && \
	PYTHONPATH=/opt/compilation/mxnet/python/:$$PYTHONPATH \
	/usr/bin/time -v \
	python -u fcn_xs.py --model=fcn8s --prefix=model_pascal/FCN16s_VGG16 --epoch=27 --init-type=fcnxs # from mxnet-fcn/run_fcnxs.sh



.PRECIOUS: VOCdev%/VOC2012/ImageSets/Segmentation/train.txt VOCdev%/VOC2012/ImageSets/Segmentation/val.txt \
 VOCtrainval_11-May-2012.tar \
 mxnet-fcn/VGG_FC_ILSVRC_16_layers-0074.params mxnet-fcn/VGG_FC_ILSVRC_16_layers-symbol.json \
 mxnet-fcn/FCN8s_VGG16-0019.params \
 mxnet-fcn/model_pascal/FCN32s_VGG16-0050.params mxnet-fcn/model_pascal/FCN16s_VGG16-0050.params mxnet-fcn/model_pascal/FCN8s_VGG16-0050.params \
 mxnet-fcn/model_pascal/FCN32s_VGG16-symbol.json mxnet-fcn/model_pascal/FCN16s_VGG16-symbol.json mxnet-fcn/model_pascal/FCN8s_VGG16-symbol.json \
 caffe-fcn/fcn-8s/fcn-8s-pascalcontext.caffemodel
