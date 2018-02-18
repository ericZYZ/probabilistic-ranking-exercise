load tennis_data

M = size(W,1);            % number of players
N = size(G,1);            % number of games in 2011 season 

psi = inline('normpdf(x)./normcdf(x)');
lambda = inline('(normpdf(x)./normcdf(x)).*( (normpdf(x)./normcdf(x)) + x)');

pv = 1.2;            % prior skill variance (prior mean is always 0)

% initialize matrices of skill marginals - means and precisions
Ms = nan(M,1); 
Ps = nan(M,1);

% initialize matrices of game to skill messages - means and precisions
Mgs = zeros(N,2); 
Pgs = zeros(N,2);

% allocate matrices of skill to game messages - means and precisions
Msg = nan(N,2); 
Psg = nan(N,2);

% 
% mean_x = linspace(1,50,50);
% %nadal_m = zeros(50);
% 
% pre_x = linspace(1,50,50);
%nadal_pre = zeros(50);

nadal_m = zeros(1,55);
monaco_m = zeros(1,55);

nadal_pre = zeros(1,55);
monaco_pre = zeros(1,55);

nadal_var = zeros(1,55);
monaco_var = zeros(1,55);

Ms_old = zeros(M,1);
Ps_old = zeros(M,1);

for iter=1:100
    
    if iter > 1
        Ms_old = Ms;
        Ps_old = Ps;
    end
    
  
  % (1) compute marginal skills 
  for p=1:M
    % precision first because it is needed for the mean update
    Ps(p) = 1/pv + sum(Pgs(G==p)); 
    Ms(p) = sum(Pgs(G==p).*Mgs(G==p))./Ps(p);
  end
 

  % (2) compute skill to game messages
  % precision first because it is needed for the mean update
  Psg = Ps(G) - Pgs;
  Msg = (Ps(G).*Ms(G) - Pgs.*Mgs)./Psg;
    
  % (3) compute game to performance messages
  vgt = 1 + sum(1./Psg, 2);
  mgt = Msg(:,1) - Msg(:,2); % player 1 always wins the way we store data
   
  % (4) approximate the marginal on performance differences
  Mt = mgt + sqrt(vgt).*psi(mgt./sqrt(vgt));
  Pt = 1./( vgt.*( 1-lambda(mgt./sqrt(vgt)) ) );
    
  % (5) compute performance to game messages
  ptg = Pt - 1./vgt;
  mtg = (Mt.*Pt - mgt./vgt)./ptg;   
    
  % (6) compute game to skills messages
  Pgs = 1./(1 + repmat(1./ptg,1,2) + 1./Psg(:,[2 1]));
  Mgs = [mtg, -mtg] + Msg(:,[2 1]);
  
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   nadal_m(iter) = Ms(1);
%   nadal_pre(iter) = Ps(1);
%   
%   means(:,iter)=Ms; 
%   covs(:,iter)=1./Ps; 
%   
%   if iter >1 
%     delta_mean = abs(Ms(1) - nadal_m(iter-1));
%     delta_pre = abs(Ps(1) - nadal_pre(iter-1));
%     
%     if (delta_mean < 0.001)&&(delta_pre < 0.001)
%         ite_final = iter;
%         break;
%     end  
%   end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  means(:,iter)=Ms; 
  covs(:,iter)=1./Ps; 
  
  nadal_m(iter) = Ms(1);
  nadal_pre(iter) = Ps(1);
  nadal_var(iter) = 1./Ps(1);
  
  monaco_m(iter) = Ms(2);
  monaco_pre(iter) = Ps(2);
  monaco_var(iter) = 1./Ps(2);
 
  if iter >1
      diff_mean = norm(Ms_old - Ms);
      diff_pre = norm(Ps_old - Ps);
     
      if (diff_mean < 0.001) && (diff_pre < 0.001)
          ite_final = iter;
          break;
      end
  end
  
  
  
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

end

win_mp = zeros(1,M);
for p = 1:M
    for q = 1:M
        if p ~= q
            new_mean = means(p,iter) - means(q,iter);
            new_cov = covs(p,iter) + covs(q,iter);
            %re2(i,j) = 1 - normcdf(0,new_mean,sqrt(new_cov+1));
            win_mp(p) = win_mp(p) + normcdf(new_mean,0,sqrt(new_cov+1));
        end
    end
end
win_mp = win_mp/(M-1);

[kk, ii_mp] = sort(win_mp, 'descend');

np = 107;
barh(kk(np:-1:1))
set(gca,'YTickLabel',W(ii_mp(np:-1:1)),'YTick',1:np,'FontSize',4)
axis([0 1 0.5 np+0.5])
title('Ranking based on message passing','FontSize',10);
xlabel('Average winning probability','FontSize',8)

[kk, ii] = sort(means(:,ite_final), 'descend');

DN_m = means((ii(1)),ite_final);
FR_m = means((ii(2)),ite_final);
NR_m = means((ii(3)),ite_final);
MA_m = means((ii(4)),ite_final);

means_4(1) = DN_m;
means_4(2) = NR_m;
means_4(3) = FR_m;
means_4(4) = MA_m;

DN_cov = covs((ii(1)),ite_final);
FR_cov = covs((ii(2)),ite_final);
NR_cov = covs((ii(3)),ite_final);
MA_cov = covs((ii(4)),ite_final);

covs_4(1) = DN_cov;
covs_4(2) = NR_cov;
covs_4(3) = FR_cov;
covs_4(4) = MA_cov;

for i = 1:4
    for j = 1:4
        if i == j
            re(i,j) = nan;
        else
            new_mean = means_4(i) - means_4(j);
            new_cov = covs_4(i) + covs_4(j);
            re(i,j) = 1 - normcdf(0,new_mean,sqrt(new_cov));
        end
    end
end

for i = 1:4
    for j = 1:4
        if i == j
            re2(i,j) = nan;
        else
            
            new_mean = means_4(i) - means_4(j);
            new_cov = covs_4(i) + covs_4(j);
            %re2(i,j) = 1 - normcdf(0,new_mean,sqrt(new_cov+1));
            re2(i,j) = normcdf(means_4(i) - means_4(j),0,sqrt(new_cov+1));
            %re2(i,j)=normcdf((means_4(i)-means_4(j))/(sqrt(1+covs_4(i)+covs_4(j))));
        end
    end
end

% figure
% p1 = plot((1:iter), nadal_m, (1:iter), nadal_pre)
% legend('Mean','Precision');
% set(p1(1),'linewidth',1.5);
% set(p1(2),'linewidth',1.5);

figure
subplot(1,2,1)
p1 = plot((1:iter), nadal_m, (1:iter), monaco_m);
title('Skill mean, 55 iterations');
xlim([0 iter])
%ylim([0 1.6])
set(p1(1),'linewidth',1.5);
set(p1(2),'linewidth',1.5);
legend('Nadal','Monaco');

subplot(1,2,2)
p2 = plot((1:iter), nadal_var, (1:iter), monaco_var);
title('Skill variance, 55 iterations');
xlim([0 iter])
set(p2(1),'linewidth',1.5);
set(p2(2),'linewidth',1.5);
legend('Nadal','Monaco');

