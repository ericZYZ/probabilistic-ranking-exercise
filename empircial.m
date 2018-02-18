load tennis_data
load ep

M = size(W,1);            % number of players
N = size(G,1);            % number of games in 2011 season 

for i=1:107
    P(i)=sum(G(:,1)==i)/(sum(G(:,1)==i)+sum(G(:,2)==i));
end


[kk, ii] = sort(P, 'descend');

count = 0;
for tt = 1:M
    if ii(tt) ~= ii_mp(tt)
        count = count +1;
    end
end

np = 107;
barh(kk(np:-1:1))
set(gca,'YTickLabel',W(ii(np:-1:1)),'YTick',1:np,'FontSize',4.5)
axis([0 1 0.5 np+0.5])
title('Ranking based on empircial game outcome','FontSize',10);
xlabel('Empircial winning probability','FontSize',8)
%ylabel('Received power (dB)','FontSize',24)