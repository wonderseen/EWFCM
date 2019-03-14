function [W, U, out_label, center, offset] = EWFCM(data, c, m, gamma, iter_totall)  
% Refference: Fuzzy Clustering With the Entropy of Attribute Weights
% Author:     Jin Zhou, Long Chen, et al.
% Link:       http://www.sciencedirect.com/science/article/pii/S0925231216003155
% Reproducing codes by wonderseen
%
%% PARAMETER DESCRIPTION
%% Input£º 
% use_start_U_flag:  (int)   just make it 0
%       data      :  (float) original data matrix by n*s, where n be the num of data points, s be the number of feature 
%        c        :  (int)   number of clusters
%        m        :  (float) fuzzy coefficients
%    iter_totall  :  (int)   end of the iteration   
% 

%% Output£º
%         W :  (float) weight matrix
%         U :  (float) membership degree matrix by c*n, Uij means the membership degree of jth point to the ith cluster
% out_label :  (int)   points with the same label belong to the same cluster
%    center :  (float) cluster matrix by c*s
%     gamma :  (float) the coefficient controlling Entropy of Attribute Weights

%% Demo on a non-spherical dataset goes as:
% V = [2 -4; 0 0];
% x1 = mvnrnd(V(:,1), [5 0; 0 0.1], 200);
% x2 = mvnrnd(V(:,2), [0.1 0; 0 5], 200);
% c = 2;
% data = [x1; x2];
% max_iteration = 50;
% m = 2;
% gamma = 0.004;
% [W, U, center, out_label] = EWFCM(data, c, m, gamma, max_iteration);

%% initial 
COLOR = ['r','g','b','m','k']; 
COLOR_CONV = ['r-','g-','b-','m-','k-']; 
[n, s] = size(data);
iter = 1;   
distance(c,n) = 0;  
center(c,s) = 0; 
center_before(c,s) = 0;

%%%%%%%%%%%%%%% initial U 
U = rand(c, n);  
temp = sum(U,1);
for j=1:n 
    U(:,j) = U(:,j)./temp(j); 
end
temp_U(c,n) = 0;

%%%%%%%%%%%%%%% initial temp variables
W = rand(c,s);
WXC = rand(c,n);
WE_num = rand(c,s);

%%%%%%%%%%%%%%% training loop
while( iter<iter_totall )  
    center_before = center;
    iter = iter + 1 ;
    %% update clusters C(t)
    Um = U.^m;
    center= Um*data ./ (sum(Um,2)*ones(1,s)); 

    %% update W
    for i=1:c
        for M=1:s
            temp = 0;
            for j=1:n
                temp = temp + Um(i,j)^m*(data(j,M)-center(i,M))^2;
            end
            WE_num(i,M) = exp(-gamma*temp);
        end
        for M=1:s
            W(i,M)=WE_num(i,M)/(WE_num(i,:)*ones(s,1));
        end
    end
    
    %% update U
    for i=1:c
        for j=1:n
            WXC(i,j) = 0;
            for M = 1:s        
                WXC(i,j) = WXC(i,j) + W(i,M)*(data(j,M)-center(i,M))^2;
            end
        end
    end
    for i=1:c
        for j=1:n
            temp = 0.0;
            for h=1:c
                temp = temp + (WXC(i,j)/(WXC(h,j)))^(1/(m-1));
            end
            U(i,j) = 1/temp;
        end
    end
    
    %% union U
    temp_U = sum((U),1);
    for j=1:n 
        U(:,j) = (U(:,j))./temp_U(j);
    end

    offset =  norm(center_before-center, inf);
    if offset< 1e-10
        break;
    end
end  

%%%%%%%%%%%%%% 4.get lable
out_label = zeros(n,1);
for j=1:n
    [ val, i ] = max(U(:,j));
    out_label(j) = i;
end

%%%%%%%%%%%%%%% 5. if 2D data, draw out the belonging results
if s==2
    fig = figure(3);
    for j=1:n
        [ val, i ] = max(U(:,j));
        scatter(data(j,1),data(j,2),4,COLOR(i));
        hold on;
    end

    %%%%%%%%%%%%%%% 5. draw out the convhull
    for i=1:c
        kind_i_args = find(out_label == i);
        single_sort = data(kind_i_args,:);

        if size(data,2)==2
            if size(single_sort,2)~=0
                k = convhull( single_sort );
                cx = single_sort( k,1 );
                cy = single_sort( k,2 );
                plot( cx, cy, COLOR_CONV(i));
                hold on;
            end
        end
    end
    
   %%%%%%%%%%%%%%%% Flexible color for VIS
    DRAW_ISOHYPSE = 1;
    if DRAW_ISOHYPSE
        if c == 2
            CLR = [U(1,:)' U(2,:)' 0.5*ones(n,1)]
        else
            CLR = [U(1,:)' U(2,:)' U(3,:)'];
        end
        scatter(data(:,1),data(:,2),100,CLR,'.');  
    end
    
    %%%%%%%%%%%%%%% 6. print the clusters
    for i=1:c
        plot([center([i],1)],[center([i],2)],'*','color',COLOR_CONV(1));
        hold on;
    end
    title('EWFCM');
    xlabel('1st Dimension'); 
    ylabel('2nd Dimension'); 
    hold off;
    print(fig,'-dpng',strcat('EWFCM','.bmp'));
end
