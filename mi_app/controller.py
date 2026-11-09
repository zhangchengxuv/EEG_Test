# -*- coding: utf-8 -*-
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog
import time
import json
from PyQt5.QtCore import QThreadPool
import numpy as np
import subprocess
# from PyQt5.QtCore import QProcess

from serialx import SerialX
import constantValues as cv
from lslSender import LSLSender
import debugPrinter as dp
from worker import Worker

from lslReceiver import LSLReceiver
from edfSaver import EDFSaver
# from generateModelX import GenerateModelX
# from miPredictor import MIPredictor
from dSPx import DSPx
# from curctrl_mu_calculator import CurCtrlCalculator
# from curctrl_classifier import CurCtrlClassifier
from curvesFormAlp import CurvesFormAlp
import os

# from gm_detect_alpha import EEGFileRead
# from offline_analysis.alpha_calibration import calibrate_from_edf
from PyQt5.QtCore import QProcess
import sys

from alpha_detector import AlphaRealtimeDetector
from alpha_detector import AlphaStateMachine

from data_aggregator import BlockAggregator

with open('task_markers.json', 'r') as file:
    markers = json.load(file)

usr_config_file = os.path.join(os.path.dirname(__file__),"ui_config", "user_config.json")

class Controller():
    def __init__(self,mainwindow,curForm):

        self.mw = mainwindow
        self.lsl = LSLSender('dummy')
        self.cf = curForm
        self.fs = EDFSaver('./data')
        self.current_file_name = None

        self.lslRcv = LSLReceiver()
        self.dsp = DSPx(cv.CH_NUM)
        # self.gm = GenerateModelX('mi')
        # self.gm_c = GenerateModelX('cc')

        self.mp = None
        self.serial = SerialX("dummy")
        # self.ccc = None

        # self.cfa = CurvesFormAlp()

        with open(usr_config_file, 'r') as file:
            self.usr_config_json = json.load(file)

        # with open(cv.Json_file_name, 'r') as file:
        #     self.usr_config_json = json.load(file)

        # self.mw.new_mac(self.usr_config_json[cv.JSON_MAC_KEY_STR])

        self.timer_ack = QtCore.QTimer()
        self.timer_ack.setInterval(1000)
        self.timer_ack.timeout.connect(self.ack_timer_handler)
        self.timer_ack.start()

        self.timer_set_mac = QtCore.QTimer()
        self.timer_set_mac.setInterval(10000)
        self.timer_set_mac.timeout.connect(self.send_mac_to_recv)
        
        # self.timer_search_inlet = QtCore.QTimer()
        # self.timer_search_inlet.setInterval(1000)
        # self.timer_search_inlet.timeout.connect(self.ack_timer_handler)

        self.mw.evt_win.connect(self.win_evt)
        # self.mw.evt_win_data.connect(self.win_evt_data)

        self.serial.evt_com_list.connect(self.serial_com_list)
        self.serial.evt_serial_cmd.connect(self.serial_evt_cmd)
        self.serial.find_serial_ports()

        self.packet_counter=0
        self.threadpool = QThreadPool()

        self.last_pkn=None

        self.lslRcv.evt_lslRcv.connect(self.lslRcv_new_data)
        self.p = None  # Default empty value.
        
        self.evt_serial_dataConnected = False

        self.serial.open_serial(self.usr_config_json[cv.JSON_COM_KEY_STR])


        self.mw.set_combox_item(self.usr_config_json[cv.JSON_COM_KEY_STR])

        self.start_time = time.time()

        self.psyco_marker_counter = 0

        # self.ad = EEGFileRead()  #alpha_detector
        params = np.load("alpha_calibration_params.npz", allow_pickle=True)
        self.ad = AlphaRealtimeDetector.from_npz(params)  #alpha_detector

        step_sec  = float(params["step"])
        alpha_thr = float(params["alpha_thr"])
        low_thr   = float(params["low_thr"])
        # fs         = float(params["fs"])
        # win_length = float(params["win_length"])

        self.ad_on = False
        self.sm = AlphaStateMachine(
            step_sec=step_sec,
            alpha_thr=alpha_thr,
            low_thr=low_thr,


            hysteresis_ratio=0.05,
            close_min_sec=1.5,
            open_min_sec=1.0,
            artifact_grace_sec=2.0,
            keep_last_on_artifact=True,
        )
        self.aggregator = BlockAggregator(block_size=int(params["step"] * params["fs"]))


    def __del__(self):
        dp.dpt('del Controller')




    def lslRcv_new_data(self,inlet_name,ts,arr):
        if (inlet_name=='mi_acc'):            
            self.cf.deal_with_data_acc_inlet(ts,arr)
            self.fs.new_data('acc',ts,arr)
        if (inlet_name=='mi_eeg'):
            ## how do you want to use this eeg data
            self.fs.new_data('eeg',ts,arr)
            arr = self.dsp.filter(arr)
            # arr = self.dsp.filter_dummy(arr)            
            self.cf.deal_with_data_inlet(ts,arr)
            # if self.mp is not None:
            #     self.mp.new_data(arr)
            # if self.ccc is not None:
            #     self.ccc.new_data(arr)

        if (inlet_name=='hb_eeg'):
            ## how do you want to use this eeg data
            # self.fs.new_data('eeg',ts,arr)
            arr = self.dsp.filter(arr)
            # arr = self.dsp.filter_dummy(arr)            
            # self.cf.deal_with_data_inlet(ts,arr)
            # if self.mp is not None:
            #     self.mp.new_data(arr)
            # if self.ccc is not None:
            #     self.ccc.new_data(arr)
            self.fs.use_one_channel()
            self.fs.new_data_1ch('eeg',ts,arr)            

            # dp.dpt(arr.shape)

            if self.ad_on :
                if arr.shape[0] == 16:

                    tmp = np.squeeze(arr)

                    # tmp is a 1-D 16 element array
                    blocks = self.aggregator.add(tmp)

                    for eeg_block in blocks:
                        results = self.ad.process_block(eeg_block)
                        if len(results)>0:
                            dp.dpt(results)
                            r = results[0]

                            stable_state = self.sm.update(r['alpha_rel'], r["low_rel"])

                            dp.dpt(stable_state)

                            if (stable_state=='open'):
                                self.serial.write_port(cv.SERIAL_CMD_act_1_off)
                                dp.dpt('on')
                            elif (stable_state=='closed'):
                                self.serial.write_port(cv.SERIAL_CMD_act_1_on)
                                dp.dpt('off')

                        # for r in results:


                            # self.handle_alpha_result(r)

                    # results = self.ad.process_block(tmp)

            self.cf.deal_with_data_inlet(ts,arr)

        if (inlet_name[:13] == 'psycho_marker'):
            self.fs.new_data('mar',ts,arr)
            # if self.mp is not None:
            #     self.mp.new_marker(arr)

            if arr[0]==markers['trial_end']:
                self.psyco_marker_counter = self.psyco_marker_counter+1
                dp.dpt('trials: '+ str(self.psyco_marker_counter))

        # if (inlet_name[:14] == 'predict_marker'):
        #     dp.dpt('--------------')

    def stim_on_exit(self,s):
        # if s==cv.EVT_WIN_MI_TEST:
        #     print('mi_test_on_exit')
        #     return

        dp.dpt('stim_on_exit')

        self.fs.flush_data()

        # if self.mp is not None:
        #     self.mp.close_predict()
        # if self.ccc is not None:
        #     self.ccc.close_calculate()

    def get_marker_inlet(self,wanted_marker_inlets):
        tmp = wanted_marker_inlets
        for i in range(30):
            time.sleep(1)
            print('find marker_inlet...'+str(i))
            new_marker_inlets = self.lslRcv.get_additional_marker_inlet()
            for inlet in new_marker_inlets:
                if inlet in tmp:
                    tmp.remove(inlet)
            if len(tmp)==0:
                return 's'
        return 'f'

    def run_task(self,s):

        self.psyco_marker_counter = 0

        wanted_marker_inlets=[]
        if s==cv.EVT_WIN_MI_TRAIN:
            task_file = 'mi_train_task.py'
            wanted_marker_inlets = ['psycho_marker_001']
        if s==cv.EVT_WIN_EYE_OC:
            task_file = 'eyeoc.py'
            wanted_marker_inlets = ['psycho_marker_002']
        if s==cv.EVT_WIN_MI_TEST:
            task_file = 'mi_test_task.py'            
            wanted_marker_inlets = ['predict_marker_1','psycho_marker_003']
        if s==cv.EVT_WIN_CURCTRL_TRAIN:
            task_file = 'curctrl_train_task.py'            
            wanted_marker_inlets = ['psycho_marker_004']
        if s==cv.EVT_WIN_CURCTRL:
            task_file = 'curctrl_test_task.py'            
            wanted_marker_inlets = ['curctrl_marker_1','psycho_marker_005']         # def process_finished(self):

         # def process_finished(self):
            #     self.p = None
            #     print('del process')

            #QProcess do not work with the qthread, do not know why
         # if self.p is None:  # No process running.
                #     self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
                #     self.p.finished.connect(self.process_finished)  # Clean up once complete.
                #     # self.p.start("python3", [task_file])
                #     self.p.start("python", ['mi_test_task.py'])
                #     dp.dpt('EVT_WIN_MI_TEST')
          # if (self.get_marker_inlet(wanted_marker_inlets)=='s'):
        #     self.p.waitForFinished()
        #     # pass
        # else:
        #     self.mw.log_info('lab streaming layer error...')
        #     self.p.terminate() 

        proc = subprocess.Popen(['python', task_file])
        if (self.get_marker_inlet(wanted_marker_inlets)=='s'):
            proc.wait()
        else:
            self.mw.log_info('lab streaming layer error...')
            proc.terminate()

        self.stim_on_exit(s)

    def new_task(self,s):
        worker = Worker(self.run_task,s)
        worker.autoDelete()
        self.threadpool.start(worker)

    def win_evt(self,s,s2):
        if s == cv.EVT_WIN_CMD_OPEN_COM:
            if (self.serial.open_serial(s2)) :
                self.usr_config_json[cv.JSON_COM_KEY_STR] = s2
        elif s == cv.EVT_WIN_CMD_CLOSE_COM:
            self.serial.close_serial()
        elif s == cv.EVT_WIN_CMD_CON_DEV:
            self.usr_config_json[cv.JSON_MAC_KEY_STR] = s2
            self.send_mac_to_recv()
        elif s == cv.SERIAL_CMD_STOP_DISCONNECT:
            self.timer_set_mac.stop()
            self.start_time = time.time()
            # print(s)
            self.serial.write_port(s)
        elif s == cv.SERIAL_CMD_FALSH_LED:
            self.serial.write_port(s)
        elif s == cv.EVT_WIN_QUIT:
            self.serial.close_serial()
            self.timer_set_mac.stop()
            self.timer_ack.stop()


            # with open(cv.Json_file_name, "w") as outfile:
            #     json.dump(self.usr_config_json, outfile)    

            with open(usr_config_file, "w") as outfile:
                json.dump(self.usr_config_json, outfile)    

            self.cf.close_win()
            # self.cfa.close_win()

            self.threadpool.waitForDone()
            dp.dpt("exit ... ")

        # elif s==cv.EVT_WIN_GENERATW_MODEL:
        #     self.gm.show()

        # elif s==cv.EVT_WIN_GENERATW_CURCTRL_MODEL:
        #     self.gm_c.show()

        ## tasks
        # elif s==cv.EVT_WIN_MI_TEST:
        #     filePathName,_ = QFileDialog.getOpenFileName(self.mw, "Select model", "./models",("mim files(*mim)"))
        #     # filePathName = 'D:/m_proj_23/mi/mi_app/models/new_model_bv8#p!_.mim'
        #     if filePathName:
        #         if (self.new_recording()=='f'):
        #             return            
        #         self.mp = MIPredictor(8)
        #         self.mp.load_model(filePathName)
        #         self.mp.open_predict()
        #         self.new_task(s)

        elif s==cv.EVT_WIN_MI_TRAIN:
            if (self.new_recording()=='f'):
                return            
            self.new_task(s)

        elif s==cv.EVT_WIN_EYE_OC:
            if (self.new_recording()=='f'):
                return            
            self.new_task(s)
            dp.dpt('EVT_WIN_EYE_OC')

        elif s==cv.EVT_WIN_BTN_ALPHA_CALIBRATION:
            dp.dpt('EVT_WIN_BTN_ALPHA_CALIBRATION')
            # self.ad.open_file(self.current_file_name)
            # self.ad.open_file()

            # edf_path = "D:\\learning_practice\\python_dsp\\detect_alpha\\12022333_.edf"
            # edf_path = ".\\data\\eyes_open\\12172105_.edf"

            # self.proc = QProcess()
            # self.proc.readyReadStandardOutput.connect(lambda: print(self.proc.readAllStandardOutput().data().decode(errors="ignore")))
            # self.proc.readyReadStandardError.connect(lambda: print(self.proc.readAllStandardError().data().decode(errors="ignore")))
            # # self.proc.finished.connect(lambda code, status: print("finished", code, status))

            # self.proc.finished.connect(self.on_calibration_finished)

            # self.proc.start(
            #     sys.executable,
            #     [
            #         "alpha_calibration.py",
            #         # "--edf", edf_path,
            #         # "--out", ".\\alpha_pars\\alpha_calibration_params.npz",
            #         # "--fig", ".\\alpha_pars\\calibration_result.png"
            #     ]
            # )




            # self.proc.start(sys.executable, [".\\offline_analysis\\alpha_calibration.py", "--edf", edf_path])

            # proc = subprocess.Popen(['python', "test.py"])
            # proc.wait()



        elif s==cv.EVT_WIN_BTN_ALPHA_DETECTION:
            self.ad_on = not self.ad_on
            dp.dpt('EVT_WIN_BTN_ALPHA_DETECTION')




        elif s==cv.EVT_WIN_CURCTRL_TRAIN:
            if (self.new_recording()=='f'):
                return            
            self.new_task(s)
            dp.dpt('EVT_WIN_CURCTRL_TRAIN')

        elif s==cv.EVT_WIN_CURCTRL:
            dp.dpt('EVT_WIN_CURCTRL')
            filePathName,_ = QFileDialog.getOpenFileName(self.mw, "Select model", "./models",("mim files(*mim)"))
            # filePathName = 'D:/m_proj_23/mi/mi_app/models/new_model_bv8#p!_.mim'
            if filePathName:
                if (self.new_recording()=='f'):
                    return            
                # self.ccc = CurCtrlClassifier(8,128)
                # self.ccc.load_model(filePathName)
                # self.ccc.open_calculate()            
                self.new_task(s)

    def new_recording(self):
        a = self.fs.get_name()
        b = a.split('/')[-1]

        text, ok = self.mw.get_input_fileName(b)
        if ok and text:
            c = a[:a.rfind('/')+1] + text
            print(c)
            if (self.fs.make_path(c)):
                self.mw.show_error("this file has been exist!")
                return 'f'
            else:
                self.fs.setup(c,text)
                self.current_file_name = self.fs.get_current_name()
                dp.dpt(self.current_file_name)
                return 's'

        return 'f'


    def on_calibration_finished(self):
        dp.dpt("on_calibration_finished")
        # print("Calibration finished")
        # pixmap = QPixmap("calibration_result.png")
        # self.ui.labelCalibration.setPixmap(pixmap)

    def stop_recording(self):
        self.fs.flush_data()

    def send_mac_to_recv(self):
        json_tmp={cv.JSON_RECV_KEY_MAC:self.usr_config_json[cv.JSON_MAC_KEY_STR]}

        # dp.dpt(self.usr_config_json[cv.JSON_MAC_KEY_STR])

        json_tmp_str = json.dumps(json_tmp)
        self.serial.write_port(json_tmp_str)

    def serial_evt_cmd(self,cmd):
        self.mw.serial_cmd(cmd)
        if cmd == cv.EVT_SERIAL_OPEN_SUC: 
            print('open suc')

            if not self.evt_serial_dataConnected:
                self.serial.evt_serial_data.connect(self.deal_serial_data)
                self.evt_serial_dataConnected=True
                # dp.dpt('evt_serial_dataConnected')
            else:
                pass
                # dp.dpt('evt_serial_data already connected ')

            self.send_mac_to_recv()
            self.timer_set_mac.start()


    def parse_data(self,json_data):

        if (cv.JSON_RECV_KEY_PACKET_NUM) in json_data.keys():
            pkn = json_data.get(cv.JSON_RECV_KEY_PACKET_NUM)

            if(self.last_pkn is None):
                self.last_pkn = pkn
            d = pkn - self.last_pkn - 1

            # insert zeros # I do not know how to transfer NAN in lsl, so I use 0 
            if(d>0)&(d<20):
                dp.dpt('lost package')
                # print('insert zeros')
                # e = np.zeros(shape=(8,), dtype=int)
                # # e[:] = np.nan
                # a = np.zeros(shape=(4,), dtype=int)
                # # a[:] = np.nan
                # for i in range(d): #how many packet lost
                #     for i in range(10):
                #         self.lsl.send_eeg(e)
                #     self.lsl.send_acc(a)
            self.last_pkn = pkn

        if (cv.JSON_RECV_KEY_CH_NUM) in json_data.keys():
            n = json_data.get(cv.JSON_RECV_KEY_CH_NUM)
            if n=="1":
                if (cv.JSON_RECV_KEY_DATA_EEG) in json_data.keys():
                    e = json_data.get(cv.JSON_RECV_KEY_DATA_EEG)
                    arr = np.array(e)
                    if arr.size!=16:
                        dp.dpt('some thing wrong')
                        return
                    for i in range(16):
                        self.lsl.send_eeg_hb(e[i:i+1])

            elif n=="8":
                if (cv.JSON_RECV_KEY_DATA_EEG) in json_data.keys():
                    e = json_data.get(cv.JSON_RECV_KEY_DATA_EEG)
                    arr = np.array(e)
                    if arr.size!=80:
                        dp.dpt('some thing wrong')
                        return

                    for i in range(10):
                        self.lsl.send_eeg(e[i*8:i*8+8])

                if (cv.JSON_RECV_KEY_DATA_ACC) in json_data.keys():
                    a = json_data.get(cv.JSON_RECV_KEY_DATA_ACC)
                    arr = np.array(a)
                    if arr.size!=4:
                        dp.dpt('some thing wrong')
                        return
                    self.lsl.send_acc(arr)


    def deal_serial_data(self,ar):
        
        try:
            json_data = json.loads(bytearray(ar))
        except ValueError as e:
            dp.dpt('--- json error')
            dp.dpt(e)
            dp.dpt('###')
            dp.dpt(ar)

            # may not suitable
            # self.serial.write_port(cv.SERIAL_ACK_OK)
            self.serial.write_port_ack_ok()

            return

        # print("--- %s ---" % (time.time() - self.start_time))

        # self.serial.write_port(cv.SERIAL_ACK_OK)

        # dp.dpt(json_data.keys())
        self.serial.write_port_ack_ok()
        self.timer_ack.start() # like feed wdg

        if (cv.JSON_RECV_KEY_EVT) in json_data.keys():
            s = json_data.get(cv.JSON_RECV_KEY_EVT)
            self.mw.dev_evt(s)
            # dp.dpt(s)

        if (cv.JSON_RECV_KEY_MAC) in json_data.keys():
            s = json_data.get(cv.JSON_RECV_KEY_MAC)
            # dp.dpt(s)
            self.mw.new_mac(s)

        if (cv.JSON_RECV_KEY_ALPHA_AMP) in json_data.keys():
            s = json_data.get(cv.JSON_RECV_KEY_ALPHA_AMP)
            arr = np.asarray(s.split(':'),dtype=np.int32)
            # I did not send decimal fraction in the esp32, so I can use int here
            # if self.cfa.showing_state == False:
            #     self.cfa.show()
            #     self.cfa.showing_state = True

            # self.cfa.new_data(arr)

        self.parse_data(json_data)
       

    def serial_com_list(self,li):

        li_ = [x for x in li if x.description() == self.usr_config_json[cv.JSON_COM_DSC_KEY_STR]]

        self.mw.add_serial_port_to_combox(li_)

    def ack_timer_handler(self):
        # print('SERIAL_ACK_OK')
        # self.serial.write_port(cv.SERIAL_ACK_OK)
        self.serial.write_port_ack_ok()

