clear all
close all
clc

A = csvread('Identification_omni_3_1.csv');
j = 2;
Pos(1,1) = 0;
Pos(1,2) = 0;

for i = 1:120
    if rem(i,10) == 1
        Pos(j,1) = sqrt((A(i,5)-Pos(j-1,1))^2 + (A(i,6)-Pos(j-1,1))^2);
        Pos(j,2) = A(i,1)-Pos(j-1,2);
        j = j+1;
    end
end


for i = 1:j-3
    vel(i,1) = (Pos(i+1,1) - Pos(i,1)) / (Pos(i+1,2) - Pos(i,2));
end
vel(1,1) = 0;

figure(1)
plot(vel(:,1));
grid on;