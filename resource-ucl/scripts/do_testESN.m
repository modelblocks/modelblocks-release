cd(fullfile(pwd,'..','genmodel'));
addpath(fullfile(pwd,'..','esns'));
load ESN
load amateur_postags possents

minout  = 1e-3;
nrsize  = length(Win);
nrnet   = size(Win{1},3);
nrtest  = length(possents);

rawsurp_esn = cell(nrtest,nrnet);

for n = 1:nrsize
    for i = 1:nrnet
        disp([n i]);
        Aout = simESN(possents,Win{n}(:,:,i)*wordvecs,Wdr{n}(:,:,i),Wout{n}(:,:,i),minout);
        for z = 1:nrtest
            rawsurp_esn{z,i}(:,n) = -log(Aout{z});
        end
    end
end

save ESNnew
