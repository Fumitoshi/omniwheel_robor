M = csvread('Omnidata_20.csv');
S=zeros(length(M(:,1)),1);
for i=1:length(M(:,1))
    S(i,1)=M(i,5)*M(i,5)+M(i,6)*M(i,6);
end
plot(M(:,1),S(:,1))
ylim([0,0.04])