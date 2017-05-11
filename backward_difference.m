A = csvread('Identification_omni_3_1.csv');
V=zeros(length(A(:,1)),1);
for i=2:length(A(:,1))-1
    Vx = (A(i+1,2)-A(i,2))/(A(i+1,1)-A(i,1));
    Vy = (A(i+1,3)-A(i,3))/(A(i+1,1)-A(i,1));
    V(i,1) = sqrt(Vx^2+Vy^2);
end
plot(A(:,1),V(:,1))