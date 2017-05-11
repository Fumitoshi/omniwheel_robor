clear all
close all
clc

A4 = csvread('Identification_omni_4_1.csv');
B4 = csvread('Identification_omni_4_2.csv');
C4 = csvread('Identification_omni_4_3.csv');
D4 = csvread('Identification_omni_4_4.csv');


t = 0:1/30:4;
lengthA4=size(A4(:,1));
S1=zeros(lengthA4(1,1),1);
for i=1:lengthA4
    S1(i,1)=sqrt(A4(i,2)*A4(i,2)+A4(i,3)*A4(i,3));
end
lengthB4=size(B4(:,1));
S2=zeros(lengthB4(1,1),1);
for i=1:lengthB4
    S2(i,1)=sqrt(B4(i,2)*B4(i,2)+B4(i,3)*B4(i,3));
end
lengthC4=size(C4(:,1));
S3=zeros(lengthC4(1,1),1);
for i=1:lengthC4
    S3(i,1)=sqrt(C4(i,2)*C4(i,2)+C4(i,3)*C4(i,3));
end
lengthD4=size(D4(:,1));
S4=zeros(lengthD4(1,1),1);
for i=1:lengthD4
    S4(i,1)=sqrt(D4(i,2)*D4(i,2)+D4(i,3)*D4(i,3));
end
yA = S1(lengthA4/2:lengthA4,1);
figure
plot(yA(1:lengthA4/2))
title('Noisy time domain signal')
YA = fft(yA,(length(t)/2));
PyyA = YA.*conj(YA)/(length(t)/2);
fA = 1000/(length(t)/2)*(0:lengthA4/2);
figure
plot(fA,PyyA(1:lengthA4/2+1))
title('Power spectral density')
xlabel('Frequency (Hz)')
figure
plot(fA(1:10),PyyA(1:10))
title('Power spectral density')
xlabel('Frequency (Hz)')

yB = S2(lengthB4/2:lengthB4,1);
figure
plot(yB(1:lengthB4/2))
title('Noisy time domain signal')
YB = fft(yB,(length(t)/2));
PyyB = YB.*conj(YB)/(length(t)/2);
fB = 1000/(length(t)/2)*(0:lengthB4/2);
figure
plot(fB,PyyB(1:lengthB4/2+1))
title('Power spectral density')
xlabel('Frequency (Hz)')

yC = S3(lengthC4/2:lengthC4,1);
figure
plot(yC(1:lengthC4/2))
title('Noisy time domain signal')
YC = fft(yC,(length(t)/2));
PyyC = YC.*conj(YC)/(length(t)/2);
fC = 1000/(length(t)/2)*(0:lengthC4/2);
figure
plot(fC,PyyC(1:lengthC4/2+1))
title('Power spectral density')
xlabel('Frequency (Hz)')


yD = S4(lengthD4/2:lengthD4,1);
figure
plot(yD(1:lengthD4/2))
title('Noisy time domain signal')
YD = fft(yD,length(t));
PyyD = YD.*conj(YD)/(length(t)/2);
fD = 1000/(length(t)/2)*(0:lengthD4/2);
figure
plot(fD,PyyD(1:lengthD4/2+1))
title('Power spectral density')
xlabel('Frequency (Hz)')