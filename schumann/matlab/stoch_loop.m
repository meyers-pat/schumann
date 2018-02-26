function stoch_loop(paramfile, jobfile, njobs, outdir, njobs_pproc, run_stochastic_again)
addpath(genpath('/home/meyers/matapps/packages/stochastic/trunk/'));
if isstr(njobs)
        njobs = str2num(njobs)
        njobs_pproc = str2num(njobs_pproc)
end

try
        run_stochastic_again;
catch
        run_stochastic_again=0;
end
% run stochastic.m
if ~strcmp(run_stochastic_again,'false')
        for ii=1:njobs
                fprintf('RUNNING JOB %d\n', ii);
                stochastic(paramfile, jobfile, ii);
        end
end
% run post-processing
stochastic_ppsf(paramfile, jobfile, outdir, njobs_pproc, 0.2, Inf, false, false, true, 'bad.txt');
