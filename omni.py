# -*- coding: utf-8 -*-


import bluetooth
import struct
import time
import sys
import cv2
import numpy as np
import os.path
import math
import csv
import pylab as plt

# 保存先指定
f = open('Omnidata.csv', 'ab')
dataWriter = csv.writer(f)
# 速度指令部分
Velocity_reference=np.array([[0.3],[0],[0]])

# flag管理部分
flag_purple=1
run_flag=1

# タイムセットアップ
time_orange=time.time()
time_green=time.time()
time_blue=time.time()
time_purple=time.time()


# 変数等定義箇所
ini_time=0
dutyMAX=60
dutymin=12
cx=0
cy=0
theta=0
##cx0=0
##cy0=0
##cx1=0
##cy1=0
##out_time=[0.0]
##out_Vx=[0.0]
##out_Vy=[0.0]
##out_W=[0.0]

Dstate=np.array([[0],[0],[0]])
Velocity=np.array([[0],[0],[0]])
State_orange=np.array([[0.0],[0.0],[0.0]])
State_blue=np.array([[0.0],[0.0],[0.0]])
State_green=np.array([[0.0],[0.0],[0.0]])
State_purple=np.array([[0.0],[0.0],[0.0]])
State1=np.array([[0.0],[0.0],[0.0]])

# omniの距離
thres=1

# 座標変換行列の定義
# global用に定義
duty_rpm=np.array([[0.0],[0],[0],[0]])
direction=([[0],[0],[0],[0]])
# omniのm/s→rpm,Volt=6.669*m/s,rpm=2000*Volt
mps_to_rpm=np.array([[143.0,0,0,0],[0,143.0,0,0],[0,0,143.0,0],[0,0,0,143.0]])
##mps_to_rpm.dtype        # 小数を使うため必要
# omni座標系でのの(Vx,Vy,W)→各モーターが出力すべき速度(V0,V1,V2,V3)
Velocity_to_Wheel=np.array([[0,1,0.085],[1,0,0.085],[0,-1,0.085],[-1,0,0.085]])
##Velocity_to_Wheel.dtype # 小数を使うため必要

# ピクセル値をメートルに変換する行列
# pixel→m
pixel_to_m=np.array([[0.00375,0,0],[0,0.00375,0],[0,0,1]])




# 橙色の空間の範囲を指定
hsv_orange_min = np.array([169,68,96])
hsv_orange_max = np.array([7,155,176])
# 緑色の空間の範囲を指定
hsv_green_min = np.array([77,64,87])
hsv_green_max = np.array([96,130,175])
# 青色の空間の範囲を指定
hsv_blue_min = np.array([102,135,130])
hsv_blue_max = np.array([114,236,211])
# 紫色の空間の範囲を指定
hsv_purple_min = np.array([114,67,96])
hsv_purple_max = np.array([148,154,182])




# 2つ円による方向検出
# [絶対座標系でomuniの中心のx座標,絶対座標系でomuniの中心のｙ座標,絶対座標系に対するomuni座標系の角度],その色の円があるかどうかの判断のフラグ=circle(入力画像,HSV変換後の画像,方向と中心座標を取りたい色の頭文字)
def circle(im,im_hsv,colorname):
# グローバルにする理由がよくわからないがバグ処理の例としてはあがっていた
##    global cx
##    global cy
##    global cx0
##    global cy0
##    global cx1
##    global cy1
##    global theta
    global State1
    global flag
##    global im_color
    flag=1
    # 色の判別
    if colorname=="o":
        hsv_min=hsv_orange_min
        hsv_max=hsv_orange_max
        color=(10,59,216)
    elif colorname=="g":
        hsv_min=hsv_green_min
        hsv_max=hsv_green_max
        color=(89,100,40)
    elif colorname=="b":
        hsv_min=hsv_blue_min
        hsv_max=hsv_blue_max
        color=(255,0,0)
    elif colorname=="p":
        hsv_min=hsv_purple_min
        hsv_max=hsv_purple_max
        color=(60,41,104)

    # マスク画像の生成
    im_m = cv2.inRange(im_hsv,hsv_min,hsv_max,)
    im_m = cv2.medianBlur(im_m,7)
    # 膨張化
    im_m = cv2.morphologyEx(im_m, cv2.MORPH_OPEN, kernel)
    # マスク画像から指定した色の領域を抽出
    im_c = cv2.bitwise_and(im,im, mask=im_m)

    gray = cv2.cvtColor(im_c, cv2.COLOR_BGR2GRAY)                     # 入力画像をグレースケール変換
    th = cv2.threshold(gray,10,255,0)[1]                            # グレースケール画像の2値化
    # 輪郭抽出
    cnts = cv2.findContours(th,1,2)[0]                              # グレースケール画像から輪郭検出
    areas = [cv2.contourArea(cnt) for cnt in cnts]
    try:
        cnt_first = [cnts[areas.index(max(areas))]][0]                  # 複数の輪郭の中で最大の輪郭を抽出
        M = cv2.moments(cnt_first)                                      # 輪郭点から白色領域の重心を計算
        center0 = np.array([[int(M["m10"]/M["m00"])],[int(M["m01"]/M["m00"])]],dtype=np.float)
# 重心を調べたいとき使用
        """print(u"重心("+ str(cx0) +","+str(cy0) + ")")                     # 重心を表示
        cv2.circle(frame1,(cx0,cy0),5, (0,0,255), -1)                       # 重心を赤円で描く
        cv2.imshow("Show Image1",frame1)
        """
        cnt_second = [cnts[areas.index(max(areas))-1]][0]
        M = cv2.moments(cnt_second)                                     # 輪郭点から白色領域の重心を計算
        center1 = np.array([[int(M["m10"]/M["m00"])],[int(M["m01"]/M["m00"])]],dtype=np.float)
# 重心を調べたいとき使用
        """print(u"重心2("+ str(cx1) +","+str(cy1) + ")")                     # 重心を表示
        cv2.circle(frame2,(cx1,cy1),5, (0,0,255), -1)                       # 重心を赤円で描く
        cv2.imshow("Show Image2",frame2)
        """
        center=2*center0/3+center1/3
        distance =center0-center1
        if np.linalg.norm(distance)>thres:                                                                   # 二つの円の距離が閾値より大きいか判断
            x=center+np.array([-distance[1],distance[0]])
            y=center+distance
            cv2.circle(im,(center[0],center[1]),7, color, -1)                                 # 原点に色を表示
            cv2.line(im,(center[0],center[1]),(y[0],y[1]),(0,255,0), 5, 1)  # 方向表示
            cv2.line(im,(center[0],center[1]),(x[0],x[1]),(0,0,255), 5, 1)  # 方向表示
##            print "center",center

            center[1][0]=480-center[1][0]     # カメラ座標系から絶対座標系に変換

            omni_xvector=np.array([-distance[1],distance[0]])
            unitvector=omni_xvector/np.linalg.norm(omni_xvector)
            theta=math.acos(unitvector[0][0])

            if unitvector[1][0]>0:
                theta=-theta
            State1=np.array([[center[0][0]],[center[1][0]],[theta]],dtype=np.float)
            State1=pixel_to_m.dot(State1)
        else:
            flag=0          # 円検出に失敗
            pass
## im_color=im
        # 結果表示
##        cv2.imshow("Show Image",im)
    except:
        flag=0              # 円検出に失敗
    return State1,flag




# omni座標系で欲しい速度を入力して各モーターに与えるduty(rpm)と方向(正転or逆転)を出す
# [duty_rpm,direction]=calculation_duty(V=[omni座標系でのx方向の速度vx,omni座標系でのｙ方向の速度vy,omni座標系での角速度w])
def calculation_duty(V):
    global duty_rpm
    global direction
    direction=np.array([[0],[0],[0],[0]])
    # omni座標系の速度入力の値を各車輪の速度へ変換
    Wheel_Velocity=Velocity_to_Wheel.dot(V)
    for i in range(0,4):
        if Wheel_Velocity[i]==0:
            zeroflag[i][0]=0
        elif Wheel_Velocity[i]<0:
            Wheel_Velocity[i]=abs(Wheel_Velocity[i])
        else:
            direction[i]=1

    duty_rpm=(256.3*Wheel_Velocity*Wheel_Velocity-16.08*Wheel_Velocity+20.22)
    duty_rpm=zeroflag*duty_rpm

    duty_rpmMAX=max(abs(duty_rpm))
    if duty_rpmMAX>dutyMAX:
        duty_rpm=np.array([[0.0],[0],[0],[0]])
    duty_rpm=np.array(duty_rpm, dtype=np.int)
    return duty_rpm,direction


# 絶対座標系→omni座標系
# omni座標系での速度=Coordinate_tf(絶対座標系での速度=[Vx,Vy,W],絶対座標系に対するomuni座標系の角度):
def Coordinate_tf(Velocity,angle):
    global Velocity1
    x=[math.cos(angle),math.sin(angle),0]
    y=[-math.sin(angle),math.cos(angle),0]
    z=[0,0,1]
    trans=np.array([x,y,z])
    Velocity1=trans.dot(Velocity)
    return Velocity1

# 微分値を返す
# d(state[T])/dt=differential(state[T],state[T-t],t)
def differential(state,oldstate,dt):
    global Dstate
    Dstate=(state-oldstate)/dt
##    print "differential",Dstate.T
    return Dstate

def nothing(x):
    pass



if __name__ == '__main__':
    # 膨張化のためのカーネル定義
    kernel = np.ones((5,5),np.uint8)
    # カメラ映像の取得
    capture = cv2.VideoCapture(0)
    # カメラセットアップ
    ret,frame = capture.read()
    # 取得した映像をグレースケール変換
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    target_name1 = "TOSABT1-001bdc064f4e"
    target_address1 = None
    target_name2 = "TOSABT2-001bdc063236"
    target_address2 = None
    target_name3 = "TOSABT3-001bdc04acf8"
    target_address3 = None
    target_name4 = "TOSABT4-001bdc04ad04"
    target_address4 = None

    bd_addr1 = "00:1b:dc:06:4f:4e"
    bd_addr2 = "00:1b:dc:06:32:36"
    bd_addr3 = "00:1b:dc:04:ac:f8"
    bd_addr4 = "00:1b:dc:04:ad:04"

    port = 1

    s1=bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    s2=bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    s3=bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    s4=bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    print "setup"
    s1.connect((bd_addr1, port))
    print"omni1 connect Success"
    s2.connect((bd_addr2, port))
    print "omni2 connect Success"
    s3.connect((bd_addr3, port))
    print"omni3 connect Success"
    s4.connect((bd_addr4, port))
    print "omni4 connect Success"


    while True:
        ret,im= capture.read()
        # 平滑化ガウシアン
        im2 = cv2.GaussianBlur(im,(25,25),0)
        # RGB色空間からHSV色空間に変換
        im_hsv = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)




# 橙色の処理
    # 後退差分法に必要なため（速度計算用）
##        oldState_orange=State_orange
        # [State=[絶対座標系でomuniの中心のx座標,絶対座標系でomuniの中心のｙ座標,絶対座標系に対するomuni座標系の角度],その色の円があるかどうかの判断のフラグ]=circle(入力画像,HSV変換後の画像,方向と中心座標を取りたい色の頭文字)

##        State_orange,flag_orange=circle(im,im_hsv,"o")
# 時間取得と微分
##        oldtime_orange=time_orange
##        time_orange=time.time()
##        dtime=time_orange-oldtime_orange


# 緑色の処理
##        oldState_green=State_green
##        State_green,flag_green=circle(im,im_hsv,"g")
# 時間取得と微分
##        oldtime_green=time_green
##        time_green=time.time()
##        dtime=time_green-oldtime_green


# 青色の処理
##        oldState_blue=State_blue
##        State_blue,flag_blue=circle(im,im_hsv,"b")
# 時間取得と微分
##        oldtime_blue=time_blue
##        time_blue=time.time()
##        dtime=time_blue-oldtime_blue


# 紫色の処理
        oldState_purple=State_purple
        State_purple,flag_purple=circle(im,im_hsv,"p")
# 時間取得と微分
        if ini_time==0:
           ini_time=time.time()
        oldtime_purple=time_purple
        time_purple=time.time()
        dtime=time_purple-oldtime_purple
        cv2.imshow("Show Image all",im)

        if flag_purple==1 and run_flag==1:
            # 現在と過去の位置から後退差分を利用して速度を出す
            Velocity_abs_purple=differential(State_purple,oldState_purple,dtime)
    ##        print "Velocity_abs_purple",Velocity_abs_purple.T
    ##        print "q_purple",q_purple.T
    ##        print "oldq_purple",oldq_purple.T
    ##        print "dtime",dtime
            # omni座標系での速度=Coordinate_tf(絶対座標系での速度=[Vx,Vy,W],絶対座標系に対するomuni座標系の角度):
            Velocity_omni_purple=Coordinate_tf(Velocity_abs_purple,-oldState_purple[2][0])
            # [それぞれのタイヤのrpmの値,それぞれのタイヤの方向]=calculation_duty(V=[omni座標系でのx方向の速度vx,omni座標系でのｙ方向の速度vy,omni座標系での角速度w])
            duty1,direction1=calculation_duty(Velocity_reference)
            print "time",dtime

            # omni1への指令部分
            s1.send("a"+str(direction1[0][0])+str(duty1[0][0]).zfill(2)+str(direction1[1][0])+str(duty1[1][0]).zfill(2)+str(direction1[2][0])+str(duty1[2][0]).zfill(2)+str(direction1[3][0])+str(duty1[3][0]).zfill(2)+"\r")
            # omni2への指令部分
##            s2.send("a"+str(direction2[0][0])+str(duty2[0][0]).zfill(2)+str(direction2[1][0])+str(duty2[1][0]).zfill(2)+str(direction2[2][0])+str(duty2[2][0]).zfill(2)+str(direction2[3][0])+str(duty2[3][0]).zfill(2)+"\r")
            # omni3への指令部分
##            s3.send("a"+str(direction3[0][0])+str(duty3[0][0]).zfill(2)+str(direction3[1][0])+str(duty3[1][0]).zfill(2)+str(direction3[2][0])+str(duty3[2][0]).zfill(2)+str(direction3[3][0])+str(duty3[3][0]).zfill(2)+"\r")
            # omni4への指令部分
##            s4.send("a"+str(direction4[0][0])+str(duty4[0][0]).zfill(2)+str(direction4[1][0])+str(duty4[1][0]).zfill(2)+str(direction4[2][0])+str(duty4[2][0]).zfill(2)+str(direction4[3][0])+str(duty4[3][0]).zfill(2)+"\r")

            # 速度を画面上に出力
##            print "time",dtime
##            print "duty1",duty1.T,"direction1",direction1.T
##            print "Velocity",Velocity_omni_purple.T
##            print "Velocity_reference",Velocity_reference.T

# 状態をCSVファイルに書き出す際に利用
##            # 時間出力
##            out_time.extend([time_purple-ini_time])
##            # omni座標系での速度出力
##            out_Vx.extend([Velocity_omni_purple[0][0]])
##            out_Vy.extend([Velocity_omni_purple[1][0]])
##            out_W.extend([Velocity_omni_purple[2][0]])
##            # 絶対座標系での速度出力
##            out_AVx.extend([Velocity_abs_purple[0][0]])
##            out_AVy.extend([Velocity_abs_purple[1][0]])
##            out_AW.extend([Velocity_abs_purple[2][0]])
##            # 絶対座標系での状態出力
##            out_Statex.extend([State_purple[0][0]])
##            out_Statey.extend([State_purple[1][0]])
##            out_Statetheta.extend([State_purple[2][0]])



        key = cv2.waitKey(10)
        # スペースキーが押されたら終了
        if key == 32:
            # 状態をCSVファイルに書き出すための処理
##            out_time=np.array(out_time)
##            out_Vx=np.array(out_Vx)
##            out_Vy=np.array(out_Vy)
##            out_W=np.array(out_W)
##            out_AVx=np.array(out_AVx)
##            out_AVy=np.array(out_AVy)
##            out_AW=np.array(out_AW)
##            out_Statex=np.array(out_Statex)
##            out_Statey=np.array(out_Statey)
##            out_Statetheta=np.array(out_Statetheta)
##            a=np.array([out_time,out_Vx,out_Vy,out_W,out_AVx,out_AVy,out_AW,out_Statex,out_Statey,out_Statetheta])
##            dataWriter.writerows(a.T)
            break
        # スペ－スキー以外の処理
        # "s"でストップ
        elif key==83:
            run_flag=0
        # "r"でリスタート
        elif key==82:
            run_flag=1
        elif key > 0:
            pass

# コネクトアウト
s1.close()
s2.close()
s3.close()
s4.close()

f.close
# キャプチャー解放
capture.release()
cv2.waitKey(30)
# ウィンドウ破棄
cv2.destroyAllWindows()
