#include <darknet.h>
#include <ros/ros.h>
#include <yolo/yoloBBox>
#include <sensor_msgs/Image.h>

void print_yolo_detections(FILE **fps, char *id, int total, int classes, int w, int h, detection *dets)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void test_yolo(char *cfgfile, char *weightfile, float thresh)
{
    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    layer l = net->layers[net->n-1];
    set_batch_network(net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    float nms=.4;

    while(ros::ok()){
       
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net->w, net->h);
        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));

        int nboxes = 0;
        detection *dets = get_network_boxes(net, 1, 1, thresh, 0, 0, 0, &nboxes);
        if (nms) do_nms_sort(dets, l.side*l.side*l.n, l.classes, nms);

        draw_detections(im, dets, l.side*l.side*l.n, thresh, voc_names, alphabet, 20);
        save_image(im, "predictions");
        show_image(im, "predictions", 0);
        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);
        if (filename) break;
    }
	ros::spin();
}

void imageRightRectifiedCallback(const sensor_msgs::Image::ConstPtr& msg) {
    ROS_INFO("Right Rectified image received from ZED - Size: %dx%d", msg->width, msg->height);
}

void imageLeftRectifiedCallback(const sensor_msgs::Image::ConstPtr& msg) {
    ROS_INFO("Left Rectified image received from ZED - Size: %dx%d", msg->width, msg->height);
}

void main(int argc, char **argv)
{
	ros::init(argc,argv,"darknet");
	ros::NodeHandle nh;
	ros::Publisher send_coord = nh.advertise<yolo::yoloBBox> ("/bboxCoord", 10);
	ros::Subscriber subRightRectified = n.subscribe("/zed/zed_node/right/image_rect_color", 10,
                                        imageRightRectifiedCallback);
   	ros::Subscriber subLeftRectified  = n.subscribe("/zed/zed_node/left/image_rect_color", 10,
                                        imageLeftRectifiedCallback);	

	float thresh = 0.2;
	int cam_index = 0;
	int frame_skip = 0;
 	char *cfg = "/home/nimbus/catkin_ws/src/tinyyolo/cfg/yolov3-tiny-obj-test.cfg";
	char *weights = "/home/nimbus/catkin_ws/src/tinyyolo/backup/realbolts/yolov3-tiny-obj_final.weights";
    //test_yolo(cfg, weights, thresh);

	ros::spin();
	return 0;
}
