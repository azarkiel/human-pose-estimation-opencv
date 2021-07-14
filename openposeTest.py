# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2 as cv
import numpy as np
import argparse

import getopt, sys
import time, os, psutil
import Metrikas as met
import logging


def main(argv):
    mi_logger = met.prepareLog('Log_'+sys.argv[0] + '.log', logging.INFO)
    process = psutil.Process(os.getpid())

    parser = argparse.ArgumentParser()
    # parser.add_argument('--inputfile', '--input', '-i', help='Path to image or video. Skip to capture frames from camera')
    parser.add_argument('-i', '--input', '--inputfile', type=str, default='0',
                        help='Path to image or video. Skip to capture frames from camera')
    # parser.add_argument("-m", "--printmetrics", help='Para imprimir metricas en la imagen')
    parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
    parser.add_argument('--printmetrics', action="store_true", help='Se agregan las metricas a la imagen de salida')
    parser.add_argument('--noview', action="store_true", help='Se procesa el video sin mostrarlo')

    args = parser.parse_args()

    BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

    POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                  ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                  ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                  ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]


    view_option = True
    print_metrics = False
    mode = "camera"
    inWidth = args.width
    inHeight = args.height



    net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

    cap = cv.VideoCapture(args.input if args.input else 0)

    success, img = cap.read()
    p_time = 0
    num_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    frame_actual = 1
    fps_list = []
    cpu_list = []
    mem_list = []

    # while cv.waitKey(1) < 0:
    while success:
        success = False
        success, frame = cap.read()
        frame_actual += 1

        # hasFrame, frame = cap.read()
        # if not hasFrame:
        #     cv.waitKey()
        #     break
        if success:
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]

            net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
            out = net.forward()
            out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

            assert (len(BODY_PARTS) == out.shape[1])

            points = []
            for i in range(len(BODY_PARTS)):
                # Slice heatmap of corresponging body's part.
                heatMap = out[0, i, :, :]

                # Originally, we try to find all the local maximums. To simplify a sample
                # we just find a global one. However only a single pose at the same time
                # could be detected this way.
                _, conf, _, point = cv.minMaxLoc(heatMap)
                x = (frameWidth * point[0]) / out.shape[3]
                y = (frameHeight * point[1]) / out.shape[2]
                # Add a point if it's confidence is higher than threshold.
                points.append((int(x), int(y)) if conf > args.thr else None)

            for pair in POSE_PAIRS:
                partFrom = pair[0]
                partTo = pair[1]
                assert (partFrom in BODY_PARTS)
                assert (partTo in BODY_PARTS)

                idFrom = BODY_PARTS[partFrom]
                idTo = BODY_PARTS[partTo]

                if points[idFrom] and points[idTo]:
                    cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                    cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                    cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

            # t, _ = net.getPerfProfile()
            # freq = cv.getTickFrequency() / 1000
            # cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time
            fps_list.append(fps)

            if not args.noview:
                if args.printmetrics:
                    cv.imshow('OpenPose using OpenCV',
                               met.printMetrics(frame, frame_actual, num_frames, fps, round(met.Average(fps_list), 1),
                                                round(max(fps_list), 1)))
                else:
                    cv.imshow('OpenPose using OpenCV', frame)

        if cv.waitKey(1) == ord('q'):
            break

        mem_list.append(process.memory_info()[0])
        # cpu_porcentaje=process.cpu_percent()
        cpu_porcentaje = round(process.cpu_percent() / psutil.cpu_count(), 1)
        cpu_list.append(cpu_porcentaje)

        sys.stdout.write("\rFrame " + str(frame_actual) + "/" + str(int(num_frames)) + " " + str(
            round(100 * frame_actual / num_frames, 1)) + "%"
                         + "\tUsedMemory=" + str(round(process.memory_info()[0] / (1024 ** 2), 1)) + "MB"
                         + "\tUsedCPU=" + str(cpu_porcentaje) + "%")
        sys.stdout.flush()

    fps_list = [i for i in fps_list if i > 0.5]
    mem_list = [i for i in mem_list if i > 0.5]
    cpu_list = [i for i in cpu_list if i > 0.5]
    resumen = ('\nARGS=' + ' '.join(str(e) for e in sys.argv)
               + '\nPROGRAM= ' + sys.argv[0]
               + '\nFILENAME= ' + sys.argv[2]
               + '\nFPS_AVG= ' + str(round(met.Average(fps_list), 1))
               + '\nFPS_MAX= ' + str(round(max(fps_list), 1))
               + '\nFPS_MIN= ' + str(round(min(fps_list), 1))
               + '\nMEM_AVG= ' + str(round(met.Average(mem_list) / (1024 ** 2), 1)) + 'MB'  # in bytes
               + '\nMEM_MAX= ' + str(round(max(mem_list) / (1024 ** 2), 1)) + 'MB'  # in bytes
               + '\nMEM_MIN= ' + str(round(min(mem_list) / (1024 ** 2), 1)) + 'MB'  # in bytes
               + '\nCPU_AVG= ' + str(round(met.Average(cpu_list), 1)) + '%'
               + '\nCPU_MAX= ' + str(round(max(cpu_list), 1)) + '%'
               + '\nCPU_MIN= ' + str(round(min(cpu_list), 1)) + '%'
               + '\n')
    print(resumen)
    mi_logger.info(resumen)


if __name__ == "__main__":
    main(sys.argv[1:])
