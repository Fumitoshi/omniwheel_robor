# -*- coding: utf-8 -*-

# 紫色と橙色を青色に合意
# omni番号紫色を1,橙色を2

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

duty=35
# 保存先指定
f = open('Omni1data_35_1.csv', 'ab')
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
dutyMAX=99
dutymin=12
cx=0
cy=0
theta=0
##cx0=0
##cy0=0
##cx1=0
##cy1=0
out_time=[0.0]
out_Vx=[0.0]
out_Vy=[0.0]
out_W=[0.0]
out_AVx=[0.0]
out_AVy=[0.0]
out_AW=[0.0]
out_Statex=[0.0]
out_Statey=[0.0]
out_Statetheta=[0.0]


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
hsv_orange_min = np.array([171,52,116])
hsv_orange_max = np.array([8,111,141])
# 緑色の空間の範囲を指定
hsv_green_min = np.array([78,76,89])
hsv_green_max = np.array([90,134,150])
# 青色の空間の範囲を指定
hsv_blue_min = np.array([103,136,115])
hsv_blue_max = np.array([112,255,201])
# 紫色の空間の範囲を指定
hsv_purple_min = np.array([120,65,77])
hsv_purple_max = np.array([138,158,153])
## ------------------------
## BGRからRGBに変換
## ------------------------
##def bgr2rbg(im):
##    b,g,r = cv2.split(im)
##    im = cv2.merge([r,g,b])
##    return im

## ------------------------
## 結果表示
## ------------------------
##def show_result(im1,im2,im3):
##
##    graph = plt.figure()
##    plt.rcParams["font.size"]=15
##    # 入力画像
##    plt.subplot(2,2,1),plt.imshow(bgr2rbg(im1))
##    plt.title("Input Image")
##    # 出力画像
##    plt.subplot(2,2,2),plt.imshow(bgr2rbg(im2))
##    plt.title("Output Image")
##    # 2値化画像
##    plt.subplot(2,2,3),plt.imshow(im3,"gray")
##    plt.title("Gray Image")
##    plt.show()


### 背景差分
##def bg_diff(fn_bg,im_in,th,blur):
##
##    # 背景, 入力画像の取得
##    im_bg = cv2.imread(fn_bg,0);
##    # 差分計算
##    diff = cv2.absdiff(im_in,im_bg)
##    # 差分が閾値より小さければTrue
##    mask = diff < th
##    # 配列（画像）の高さ・幅
##    hight = im_bg.shape[0]
##    width = im_bg.shape[1]
##    # 背景画像と同じサイズの配列生成
##    im_mask = np.zeros((hight,width),np.uint8)
##    # Trueの部分（背景）は白塗り
##    im_mask[mask]=255
##    # ゴマ塩ノイズ除去
##    im_mask = cv2.medianBlur(im_mask,blur)
##    # エッジ検出
##    im_edge = cv2.Canny(im_mask,100,200)
##
##    return im_bg, im_in, im_mask, im_edge

### ベクトルのユークリッドノルムを返す
##def norm(x,y):
##    Value= math.sqrt(x*x+y*y,0.5)
##    return Value

### (cx1,vy1)から(cx0,cy0)への単位ベクトルを出力
### [x,y]=unitvector(cx0,cy0,cx1,cy1)
##def unitvector(cx0,cy0,cx1,cy1):
##    x=cx0 - cx1
##    y=cy0 - cy1
##    normValue=norm(x,y)
##    x=x/normValue
##    y=y/normValue
##    return x,y




# 2つ円による方向検出
# q=[絶対座標系でomuniの中心のx座標,絶対座標系でomuniの中心のｙ座標,絶対座標系に対するomuni座標系の角度],その色の円があるかどうかの判断のフラグ=circle(入力画像,HSV変換後の画像,方向と中心座標を取りたい色の頭文字)
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



### m/s→RPM(PWM)
##def tf(V):
##    Volt=V*6.669    # m/s→V
##    RPM=Volt*2000        # V→RPM(PWM)
##    return rpm

### dutyの閾値設定
##def duty_scope(rpm):
##    if rpm>dutyMAX:
##        rpm=dutyMAX
##    elif rpm<dutymin:
##        rpm=dutymin
##    return rpm

### 方向判断
##def dicision_direction(Velocity):
##    if Velocity>0:
##        direction=1
##    else:
##        direction=0
##        Velocity=(Velocity)
##    return Velocity,direction

# omni座標系で欲しい速度を入力して各モーターに与えるduty(rpm)と方向(正転or逆転)を出す
# [duty_rpm,direction]=duty(V=[omni座標系でのx方向の速度vx,omni座標系でのｙ方向の速度vy,omni座標系での角速度w])
def calculation_duty(V):
    global duty_rpm
    global direction
    direction=np.array([[0],[0],[0],[0]])
    # omni座標系の速度入力の値を各車輪の速度へ変換
    Wheel_Velocity=Velocity_to_Wheel.dot(V)
##    duty_Velocity[0]=vy+0.085*w
##    duty_Velocity[1]=-vx+0.085*w
##    duty_Velocity[2]=-vy+0.085*w
##    duty_Velocity[3]=vx+0.085*w

    duty_rpm=mps_to_rpm.dot(Wheel_Velocity)
##    duty_Velocity[0],direction=(duty_Velocity[0])
##    duty_rpm[0]=tf(duty_V[0])
##    duty_rpm[1]=tf(duty_V[1])
##    duty_rpm[2]=tf(duty_V[2])
##    duty_rpm[3]=tf(duty_V[3])
    for i in range(0,4):
        if duty_rpm[i]<0:
            duty_rpm[i]=abs(duty_rpm[i])
        else:
            direction[i]=1

        if duty_rpm[i]>dutyMAX:
            duty_rpm[i]=dutyMAX
##        elif duty_rpm[i]<dutymin:
##            duty_rpm[i]=dutymin
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
    # 閾値調整用のスライダー生成
##    cv2.namedWindow("Motion Edge")
##    cv2.createTrackbar("threshold", "Motion Edge", 60, 255, nothing)
    # カメラ映像の取得
    capture = cv2.VideoCapture(0)
    # カメラセットアップ
    ret,frame = capture.read()
    # 計算高速化のために画像サイズを1/2
##    frame = cv2.resize(frame,(frame.shape[1]/2,frame.shape[0]/2))
    # 取得した映像をグレースケール変換
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 背景画像がないときのフレームを背景画像として生成
##    if os.path.exists("bg.jpg")==False:
##        cv2.imwrite('bg.jpg',frame)
    '''#背景差分を利用する際には利用
    print"space key:recapture background"
    print"the others:quit"
    '''

    target_name1 = "TOSABT1-001bdc064f4e"
    target_address1 = None
##    target_name2 = "TOSABT2-001bdc063236"
##    target_address2 = None

    bd_addr1 = "00:1b:dc:06:4f:4e"
##    bd_addr2 = "00:1b:dc:06:32:36"

    port = 1

    s1=bluetooth.BluetoothSocket(bluetooth.RFCOMM)
##    s2=bluetooth.BluetoothSocket(bluetooth.RFCOMM)

    print "setup"
    s1.connect((bd_addr1, port))
    print"omni1 connect Success"
##    s2.connect((bd_addr2, port))
##    print "omni2 connect Success"


##    K=raw_input(">")

    while True:
##        ret,frame= capture.read()
        ret,im= capture.read()
        # 計算高速化のために画像サイズを1/2
##        im = cv2.resize(im,(im.shape[1]/2,im.shape[0]/2))
##        im_color=im
        # 計算高速化のために画像サイズを1/2
        # ラベリング処理を出力する際使用
##        frame2 = cv2.resize(frame,(frame.shape[1]/2,frame.shape[0]/2))
##        th = cv2.getTrackbarPos("threshold", "Motion Edge")
        # 取得した映像をグレースケール変換(背景差分を利用する際使用)
##        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 平滑化ガウシアン
        im2 = cv2.GaussianBlur(im,(25,25),0)
        # RGB色空間からHSV色空間に変換
        im_hsv = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)




# 橙色の処理
##        oldState_orange=State_orange
##        State_orange,flag_orange=circle(im,im_hsv,"o")
##        oldtime_orange=time_orange
##        time_orange=time.time()
##        dtime=time_orange-oldtime_orange


# 緑色の処理
##        oldState_green=State_green
##        State_green,flag_green=circle(im,im_hsv,"g")
##        oldtime_green=time_green
##        time_green=time.time()
##        dtime=time_green-oldtime_green


# 青色の処理
##        oldState_blue=State_blue
##        State_blue,flag_blue=circle(im,im_hsv,"b")
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
##        q_differential_purple=differential(q_purple,oldq_purple,dt)
        cv2.imshow("Show Image all",im)

# 1番のomniへのflag処理
##        if flag_purple==0 and flag_orange==0:
##            duty1[0]=1
##            duty1[1]=1
##            duty1[2]=1
##            duty1[3]=1
##            s1.send("a"+str(direction1[0])+str(duty1[0])+str(direction1[1])+str(duty1[1])+str(direction1[2])+str(duty1[2])+str(direction1[3])+str(duty1[3])+"\r")
        if flag_purple==1 and run_flag==1:
            # Velocity=np.array([Vx],[Vy],[W],np.float)
            Velocity_abs_purple=differential(State_purple,oldState_purple,dtime)
    ##        print "Velocity_abs_purple",Velocity_abs_purple.T
    ##        print "q_purple",q_purple.T
    ##        print "oldq_purple",oldq_purple.T
    ##        print "dtime",dtime
            Velocity_omni_purple=Coordinate_tf(Velocity_abs_purple,-oldState_purple[2][0])
            duty1,direction1=calculation_duty(Velocity_reference)
##            print "time",dtime
##            s1.send("a"+str(direction1[0][0])+str(duty1[0][0]).zfill(2)+str(direction1[1][0])+str(duty1[1][0]).zfill(2)+str(direction1[2][0])+str(duty1[2][0]).zfill(2)+str(direction1[3][0])+str(duty1[3][0]).zfill(2)+"\r")
##            print "duty1",duty1.T,"direction1",direction1.T


            s1.send("a"+"000"+"1"+str(duty)+"000"+"0"+str(duty)+"\r")

            print "Velocity",Velocity_omni_purple.T
            print "Velocity_abs",Velocity_abs_purple.T
##            print "Velocity_reference",Velocity_reference.T
            out_time.extend([time_purple-ini_time])
            out_Vx.extend([Velocity_omni_purple[0][0]])
            out_Vy.extend([Velocity_omni_purple[1][0]])
            out_W.extend([Velocity_omni_purple[2][0]])
            out_AVx.extend([Velocity_abs_purple[0][0]])
            out_AVy.extend([Velocity_abs_purple[1][0]])
            out_AW.extend([Velocity_abs_purple[2][0]])
            out_Statex.extend([State_purple[0][0]])
            out_Statey.extend([State_purple[1][0]])
            out_Statetheta.extend([State_purple[2][0]])
    ### 2番のomniへのflag処理
    ##        if flag_blue==0 and flag_orange==0:


    ##            duty2[0]=
    ##            duty2[1]=
    ##            duty2[2]=
    ##            duty2[3]=
    ##
    ##            s2.send("a"+str(direction2)+str(duty2[0])+str(direction2)+str(duty2[1])+str(direction2)+str(duty2[2])+str(direction2)+str(duty2[3])+"\r")



        key = cv2.waitKey(10)
        # スペースキーが押されたら終了
        if key == 32:
            out_time=np.array(out_time)
##            out_Vx=np.array(out_Vx)
##            out_Vy=np.array(out_Vy)
##            out_W=np.array(out_W)
            out_AVx=np.array(out_AVx)
            out_AVy=np.array(out_AVy)
            out_AW=np.array(out_AW)
            out_Statex=np.array(out_Statex)
            out_Statey=np.array(out_Statey)
            out_Statetheta=np.array(out_Statetheta)
##            a=np.array([out_time,out_Vx,out_Vy,out_W,out_AVx,out_AVy,out_AW,out_Statex,out_Statey,out_Statetheta])
            a=np.array([out_time,out_AVx,out_AVy,out_AW,out_Statex,out_Statey,out_Statetheta])
##            a=np.array([out_time,out_AVx,out_AVy,out_AW])
            dataWriter.writerows(a.T)
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
f.close
# キャプチャー解放
capture.release()
cv2.waitKey(30)
# ウィンドウ破棄
cv2.destroyAllWindows()
