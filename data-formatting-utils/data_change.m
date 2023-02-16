data=speed;


n=0;
for n=1:35040
    n
    temp= data(1:28, :);
    temp=table2array(temp);
    x(n,:)=transpose(temp);
    temp=[];
    data(1:28, :)=[];
    n=n+1;
end 

% csvwrite('speed2019modified.csv',x)