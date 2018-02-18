load tennis_data
load ep

randn('seed',27); % set the pseudo-random number generator seed


ite = 30000;
M = size(W,1);            % number of players
N = size(G,1);            % number of games in 2011 season 

pv = 0.5*ones(M,1);           % prior skill variance 
%pv = 1*ones(M,1); 

w = zeros(M,1);               % set skills to prior mean
%w = ones(M,1);
nadal = zeros(1, ite);
djokovic = zeros(1, ite);
monaco = zeros(1, ite);

nadal_v = zeros(1, ite);
monaco_v = zeros(1, ite);

nadal_m = zeros(1, ite);
djokovic_m = zeros(1, ite);
monaco_m = zeros(1, ite);

nadal_m_thin = zeros(1, 110);
monaco_m_thin = zeros(1, 110);

x = linspace(1,ite,ite);  
mixing_time = 10;
mixing_x = linspace(1,ite/mixing_time,ite/mixing_time);

mixing_DN = zeros(1, ite/mixing_time);
mixing_NR = zeros(1, ite/mixing_time);
mixing_FR = zeros(1, ite/mixing_time);
mixing_MA = zeros(1, ite/mixing_time);

counter = zeros(4);
count = 0;
samples = zeros(M, ite/mixing_time);
for i = 1:ite      

  % First, sample performance differences given the skills and outcomes
  
  t = nan(N,1); % contains a t_g variable for each game
  for g = 1:N   % loop over games
    s = w(G(g,1))-w(G(g,2));  % difference in skills
    t(g) = randn()+s;         % performace difference sample
    while t(g) < 0  % rejection sampling: only positive perf diffs accepted
      t(g) = randn()+s; % if rejected, sample again
    end
  end 
 
  
  % Second, jointly sample skills given the performance differences
  
  m = nan(M,1);  % container for the mean of the conditional 
                 % skill distribution given the t_g samples
  for p = 1:M
     % (***TO DO***) complete this line
     m(p) = (t')*((p == G(:,1)) - (p == G(:,2)));
  end
  
  iS = zeros(M,M); % container for the sum of precision matrices contributed
                   % by all the games (likelihood terms)
%   for g = 1:N
%       % (***TO DO***) build the iS matrix
%       for ind1 = 1:M
%           for ind2 = 1:M
%               if ind1==ind2
%                   iS(ind1,ind2) = iS(ind1,ind2) + ((ind2 == G(g,1)) + (ind2 == G(g,2)));
%               else
%                   iS(ind1,ind2) = iS(ind1,ind2) - (ind1 == G(g,1))*(ind2 == G(g,2)) - (ind1 == G(g,2))*(ind2 == G(g,1));
%               end
%           end       
%       end
%   end


  for g = 1:N
      iS(G(g,1), G(g,1)) = iS(G(g,1), G(g,1)) + 1;
      iS(G(g,2), G(g,2)) = iS(G(g,2), G(g,2)) + 1;
      iS(G(g,1), G(g,2)) = iS(G(g,1), G(g,2)) - 1;
      iS(G(g,2), G(g,1)) = iS(G(g,2), G(g,1)) - 1;
  end
  
  iSS = diag(1./pv) + iS; % posterior precision matrix
  % prepare to sample from a multivariate Gaussian
  % Note: inv(M)*z = R\(R'\z) where R = chol(M);
  iR = chol(iSS);  % Cholesky decomposition of the posterior precision matrix
  mu = iR\(iR'\m); % equivalent to inv(iSS)*m but more efficient
    
  % sample from N(mu, inv(iSS))
  %w = mu + iR\randn(M,1);
  
  
  if i > 1
      w_old = w;
      w = mu + iR\randn(M,1);
      w_diff = w - w_old;
      w_diff_norm = norm(w_diff);
      count = count+1;
      
    if w_diff_norm < 0.5
      
        break;
    end
  end
  
  
  
  nadal(i) = w(1);
  djokovic(i) = w(16);
  monaco(i) = w(2);
  
  nadal_m(i) = sum(nadal)/i;
  djokovic_m(i) = sum(djokovic)/i;
  monaco_m(i) = sum(monaco)/i;
  
  nadal_v(i) = var(nadal(1:i));
  monaco_v(i) = var(monaco(1:i));
  
  if (mod(i,10) == 0)
      mixing_DN(i/10) = w(16);
      mixing_NR(i/10) = w(1);
      mixing_FR(i/10) = w(5);
      mixing_MA(i/10) = w(11);
      
      samples(:,i/10)=w; 
      
      nadal_m_thin(i/10) = sum(mixing_NR)/(i/10);
    
      
      mix(1) = w(16);
      mix(2) = w(1);
      mix(3) = w(5);
      mix(4) = w(11);
     
      for fi = 1:4
          for se = 1:4
              if fi == se
                  
              else
                  counter(fi, se) = counter(fi, se) + (mix(fi) > mix(se));
              end
          end
      end
  end
  
    
end


%Question e

win = zeros(1,M);
for e1 = 1:M
    for e2 = 1:M
        if e1 ~= e2
            win(e1) = win(e1) + sum(normcdf(samples(e1,:) - samples(e2,:)))/3000;
        end
    end
end
win = win/(M-1);

[kk, ii] = sort(win, 'descend');

ct = 0;
for tt = 1:M
    if ii(tt) ~= ii_mp(tt)
        ct = ct +1;
    end
end

figure
np = 107;
barh(kk(np:-1:1))
set(gca,'YTickLabel',W(ii(np:-1:1)),'YTick',1:np,'FontSize',4)
axis([0 1 0.5 np+0.5])
title('Ranking based on Gibbs sampling','FontSize',10);
xlabel('Average winning probability','FontSize',8)

% figure
% plot((1:length(nadal_m_thin)), nadal_m_thin)
% xlim([0 110])
% 
% figure
% plot((1:length(monaco_m_thin)), monaco_m_thin)
% xlim([0 110])

figure
subplot(1,2,1)
qb1 = plot((1:ite), nadal_m, (1:ite), monaco_m);
title('Skill mean, 11000 iterations');
xlim([0 ite])
%ylim([0 1.6])
set(qb1(1),'linewidth',1.5);
set(qb1(2),'linewidth',1.5);
legend('Nadal','Monaco');

subplot(1,2,2)
qb2 = plot((1:ite), nadal_v, (1:ite), monaco_v);
title('Skill variance, 11000 iterations');
xlim([0 ite])
set(qb2(1),'linewidth',1.5);
set(qb2(2),'linewidth',1.5);
legend('Nadal','Monaco');

mixing = [mixing_DN;mixing_NR;mixing_FR;mixing_MA];
% mixing(1) = mixing_DN;
% mixing(2) = mixing_NR;
% mixing(3) = mixing_FR;
% mixing(4) = mixing_MA;

[ac1, lag1] = xcov(nadal,50,'coeff');
[ac2, lag2] = xcov(djokovic,50,'coeff');
[ac3, lag3] = xcov(monaco,50,'coeff');

figure
ac = plot(lag1,ac1,lag2,ac2,lag3,ac3);
set(ac(1),'linewidth',1.5);
set(ac(2),'linewidth',1.5);
set(ac(3),'linewidth',1.5);
xlim([-50 50])
legend('Nadal','Djokovic','Monaco');

covariance = zeros(4);
for p=1:4
    for q=1:p
        covariance_m = cov(mixing(p,:),mixing(q,:));
        covariance(p,q) = covariance_m(1,2);
        covariance(q,p) = covariance(p,q);
    end
end

figure
subplot(3,1,1)
h1 = plot(x, nadal, x, nadal_m);
set(h1(1),'linewidth',1);
set(h1(2),'linewidth',1.5);
title('Player skill samples for Nadal');
xlim([0 ite])
legend('samples','mean of samples');

subplot(3,1,2)
h2 = plot(x, djokovic, x,djokovic_m );
set(h2(1),'linewidth',1);
set(h2(2),'linewidth',1.5);
title('Player skill samples for Djokovic');
xlim([0 ite])
legend('samples','mean of samples');

subplot(3,1,3)
h1 = plot(x, monaco, x, monaco_m);
set(h1(1),'linewidth',1);
set(h1(2),'linewidth',1.5);
title('Player skill samples for Juan-Monaco');
xlim([0 ite])
legend('samples','mean of samples');



number = ite/mixing_time;

%Marginal means
mle_mean_DN = sum(mixing_DN)/(number);
mle_mean_NR = sum(mixing_NR)/(number);
mle_mean_FR = sum(mixing_FR)/(number);
mle_mean_MA = sum(mixing_MA)/(number);

mle_mean(1) = mle_mean_DN;
mle_mean(2) = mle_mean_NR;
mle_mean(3) = mle_mean_FR;
mle_mean(4) = mle_mean_MA;


%Marginal variances
var_DN = 0;
var_NR = 0;
var_FR = 0;
var_MA = 0;

for in = 1:number
    var_DN = var_DN + (mle_mean_DN - mixing_DN(in)).^2;
    var_NR = var_NR + (mle_mean_NR - mixing_NR(in)).^2;
    var_FR = var_FR + (mle_mean_FR - mixing_FR(in)).^2;
    var_MA = var_MA + (mle_mean_MA - mixing_MA(in)).^2;
end

mle_var_DN = var_DN/(number - 1);
mle_var_NR = var_NR/(number - 1);
mle_var_FR = var_FR/(number - 1);
mle_var_MA = var_MA/(number - 1);

mle_var(1) = mle_var_DN;
mle_var(2) = mle_var_NR;
mle_var(3) = mle_var_FR;
mle_var(4) = mle_var_MA;


%Construct table1
for f = 1:4
    for s = 1:4 
        if f == s
           re1(f,s) = nan;
        else
           new_mean = mle_mean(f) - mle_mean(s);
           new_cov = mle_var(f) + mle_var(s)-2*covariance(f,s);
           re1(f,s) = 1 - normcdf(0,new_mean,sqrt(new_cov));
        end
    end
end


%Construct table2

for i1 = 1:4
    for i2 = 1:4
        re2(i1,i2) = counter(i1,i2)/number;
    end
end


%plot (mixing_x, mixing);