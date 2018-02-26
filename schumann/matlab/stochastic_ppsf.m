% postProcessScriptFull - script for post processing the data using the same
% inputs as stochastic.m, plus some additional parameters used only for
% post-processing.
%
% This script is intended to be run after first generating individual results
% for each job. Concatenation of individual job files is done automatically
% as part of this script using Unix calls to 'cat'. The complete set of steps
% needed to run the stochastic pipeline is:
%
% 1. Generate list of segments using SegWizard and save to the jobs file
% 2. Edit parameters file if necessary
% 3. Run stochastic_pipe.tclsh to create frame cache for all jobs
% 4. Run stochastic.m
% 5. Run postProcessScriptFull
%
% Inputs:
%   paramsFile - parameters file
%   jobsFile - list of science segments
%   outputDir - directory where post-processing results will go
%   dSigmaCut - max allowable relative difference between naive and theoretical
%               sigmas
%   largeSigmaCutoff
%   doRenormalize
%   modifyFilter
%   displayResults - set to "false" to suppress display of plots
%   badGPSTimesFile - file to which you wish to save badGPSTimes (optional)
%   printFormats - a cellarray listing format(s) to print plots as. For example, the default is
%     { 'epsc2', 'png' }, which causes all plots to be printed in colour .eps and .png format.
%     Some formats such as .png take a long time to print.
%     A single format can be expressed as a string or cellarray eg. 'epsc2' or { 'epsc2' }.
%     Printing can be completely suppressed by providing an empty list {}.
%
% NOTE: when running the compiled postProcessScriptFull it is not possible to
% use a modified filter - modifyFilter must always be 0.
function postProcessScriptFull(paramsFile, jobsFile, outputDir, numJobs, dSigmaCut, largeSigmaCutoff, ...
                               doRenormalize, modifyFilter, displayResults, applyBadGPSTimes, badGPSTimesFile)%, printFormats)

  datestamp  = datestr(now, 31);

  % Optional parameter to disable plotting entirely
  if (nargin < 12)
    printFormats = { 'epsc2', 'png' };
  end;

  % If just a single format, make into a cellarray
  if (~iscell(printFormats))
    printFormats = { printFormats };  
  end;
  
  if (nargin < 11)
    badGPSTimesFile = '';
  end;

  if (ischar(displayResults))
    displayResults = str2num(displayResults);
  end;

  if (ischar(dSigmaCut))
    dSigmaCut = str2num(dSigmaCut);
  end;

  if (ischar(largeSigmaCutoff))
    largeSigmaCutoff = str2num(largeSigmaCutoff);
  end;

  if (ischar(doRenormalize))
    doRenormalize = str2num(doRenormalize);
  end;

  if (ischar(modifyFilter))
    modifyFilter = str2num(modifyFilter);
  end;

  if (~displayResults)
    set(0, 'DefaultFigureVisible', 'off');
    set(0, 'DefaultAxesVisible', 'off');
  end;

  if (~exist(outputDir, 'dir'))
    error(sprintf('Output dir %s does not exist', outputDir));
  end;
  applyBadGPSTimes

  if ~(applyBadGPSTimes == 0 | applyBadGPSTimes == 1 | isa(applyBadGPSTimes, 'char'))
    s = sprintf('Please specify applyBadGPSTimes as 0 (for no cuts), 1 (for delta sigma cuts applied on the fly), or a filename of bad GPS times to apply.');
    error(s);
  end

  if isa(applyBadGPSTimes,'char')
    if ~exist(applyBadGPSTimes,'file')
      s=sprintf('Bad GPS times file \''%s\'' does not exist', applyBadGPSTimes);
      error(s);
    end
  end

  sigmaMinRatio = 1 - dSigmaCut;
  sigmaMaxRatio = 1 + dSigmaCut;

  % These appear to be the same for all jobs
  figureNumber = 1;
  
  % TODO: what if there are multiple trials?
  trial = 1;

  % read in params structure from a file
  params = setStochasticParams(paramsFile, jobsFile, 1);
  checkParamsStochastic(params);

% Temporarily comment this out to match how the original stochastic post-processing works, it ignores combine data
%  if (params.doCombine)
%    combStr = '_combined';
%  else
%    combStr = '';
%  end;
  if (params.doCombine)
    warning('doCombine option is ignored in post-processing, only uncombined results are used.');
    params.doCombine = false;
  end;
  combStr = '';

  if (params.writeOutputToMatFile)
    % don't use combined data for naive sigmas
    naivesigmasfileprefix = [ params.outputFilePrefix '.job' ];
    statsfileprefix = [ params.outputFilePrefix combStr '.job' ];
    spectrafileprefix = [ params.outputFilePrefix combStr '.job' ];
    sensintsfileprefix = [ params.outputFilePrefix combStr '.job' ];
    fileSuffix = '.mat';
  else
    % don't use combined data for naive sigmas
    naivesigmasfileprefix = [ params.outputFilePrefix '_naivesigmas.job' ];
    statsfileprefix = [ params.outputFilePrefix combStr '_ccstats.job' ];
    spectrafileprefix = [ params.outputFilePrefix combStr '_ccspectra.job' ];
    sensintsfileprefix = [ params.outputFilePrefix combStr '_sensints.job' ];
    fileSuffix = [ '.trial' num2str(trial) '.dat' ];
  end;

  % Prefix for output from post-processing
  outputFileNamePrefix = [ outputDir '/' params.ifo1 params.ifo2 ];
  figureLegend = [ params.ifo1 '-' params.ifo2 ];

  % read in job start time and duration from a file so we can count jobs and segments
  [startTimes, jobDurations] = readJobsFile(jobsFile, params.jobsFileCommentStyle);
  if numJobs == -1
      numJobs = length(startTimes);
  end
  numSegments = sum(jobDurations(1:numJobs))/params.segmentDuration;

  numPoints1 = params.segmentDuration*params.resampleRate1;
  numPoints2 = params.segmentDuration*params.resampleRate2;
  window1 = tukeywin(numPoints1, params.segmentDuration*params.resampleRate1);
  window2 = tukeywin(numPoints2, params.segmentDuration*params.resampleRate2);

  % Write datestamp
  fid = fopen([ params.outputFilePrefix '_datestamp.txt' ], 'w');
  fprintf(fid, '%s\n', datestamp);
  fclose(fid);

  % Concatenate file names from all jobs into a cell array
  concatNaiveSigmas = {};
  concatCCStats = {};
  for job = 1:numJobs
    concatNaiveSigmas{job} = [ naivesigmasfileprefix num2str(job) fileSuffix ];
    concatCCStats{job} = [ statsfileprefix num2str(job) fileSuffix ];
  end;

  % TODO: This needs to be regression tested
  if (params.writeNaiveSigmasToFiles)

    % relative sigma cuts
    if (applyBadGPSTimes == 0)
      sigmaMinRatio = 0;
      sigmaMaxRatio = Inf;
      fprintf('Not applying delta sigma cut or user cuts.\n')
    end;

    [gpsTimes, sigmas, naiveSigmas, badGPSTimes_abssigmacut, ...
     goodGPSTimes, goodSigmas, goodNaiveSigmas] = ...
        compareSigmasFromFile(concatNaiveSigmas, sigmaMinRatio, sigmaMaxRatio);
    
    % user cuts
    if isa(applyBadGPSTimes,'char')
      fprintf('Applying user cuts, but not delta sigma cuts.\n');
      badGPSTimes_abssigmacut=load(applyBadGPSTimes);
      goodGPSTimes=gpsTimes(find(~ismember(gpsTimes,badGPSTimes_abssigmacut)));
      goodSigmas=sigmas(find(~ismember(gpsTimes,badGPSTimes_abssigmacut)));
      goodNaiveSigmas=naiveSigmas(find(~ismember(gpsTimes,badGPSTimes_abssigmacut)));
    end

    % absolute sigma cuts
    badGPSTimes_largesigmacut = largeSigmas(concatNaiveSigmas, largeSigmaCutoff);

    % union cuts
    badGPSTimes = [ badGPSTimes_abssigmacut; badGPSTimes_largesigmacut ];
    badGPSTimes = sort(unique(badGPSTimes));

    % report results
    if (displayResults)
      fprintf(1, [datestamp '\n']);
      if (applyBadGPSTimes == 0 | applyBadGPSTimes == 1)
        fprintf(['Number of segments lost to delta sigma cut = %d,' ...
          ' fraction = %g\n'], ...
          length(badGPSTimes_abssigmacut), ...
          length(badGPSTimes_abssigmacut)/numSegments);
      elseif isa(applyBadGPSTimes, 'char')
        fprintf(['Number of segments lost to user-specified cut = %d,' ...
          ' fraction = %g\n'], ...
          length(badGPSTimes_abssigmacut), ...
          length(badGPSTimes_abssigmacut)/numSegments);
      end;
      fprintf(['Number of segments lost to abs sigma cut = %d,' ...
               ' fraction = %g\n'], ...
              length(badGPSTimes_abssigmacut), ...
              length(badGPSTimes_abssigmacut)/numSegments);
      fprintf(['Number of segments lost to large sigma cut = %d,' ...
               ' fraction = %g\n'], ...
              length(badGPSTimes_largesigmacut), ...
              length(badGPSTimes_largesigmacut)/numSegments);
      fprintf('Total number of segments lost to cuts = %d, fraction = %g\n', ...
              length(badGPSTimes), length(badGPSTimes)/numSegments);
      fprintf('Total time remaining after cuts = %5.1f days\n', ...
              (numSegments-length(badGPSTimes))*params.segmentDuration/(3600*24));
    end;
    clear badGPSTimes_abssigmacut badGPSTimes_largesigmacut;

    % save results 
    if length(badGPSTimesFile) > 0
      % in mat-file format if possible
      if strcmp(badGPSTimesFile(length(badGPSTimesFile)-3:end),'.mat')
        save(badGPSTimesFile,'badGPSTimes','-v6')
      else
        fid = fopen(badGPSTimesFile,'w');
        for ii = 1:length(badGPSTimes)
          fprintf(fid, '%d\n', badGPSTimes(ii));
        end;
       fclose(fid);
      end;
    else
      warning('No badGPSTimesFile file provided; not saving badGPSTimes.');
    end;

  else
    warning('Naive sigmas not written - sigma cuts will not be performed.');
    badGPSTimes = [];
  end; % if params.writeNaiveSigmasToFiles

  % Ok, done with pre-post-processing.  Let's start the post-processing.
  % For keeping plots straight, figure #s are as numbered in Vuk's elog post at
  % http://ldas-sw.ligo.caltech.edu/ilog/pub/ilog.cgi?group=stochastic&date_to_view=08/31/2005&anchor_to_scroll_to=2005:08:31:12:10:03-vmandic

  % Figure 1 (+ two new figures)
  runningPointEstimate(concatCCStats, badGPSTimes, params.resampleRate1, ...
                       params.segmentDuration, outputFileNamePrefix, figureLegend, ...
                       params.doOverlap, displayResults, printFormats);
  fprintf('** Done runningPointEstimate\n');

  % Figures 2-7
  [ptEstimate, errorBar, combinedPtEstInt, combinedSensInt, numSegmentsTotal] = ...
      combineResultsFromMultipleJobs(statsfileprefix, spectrafileprefix, ...
                                     sensintsfileprefix, fileSuffix, numJobs, params.segmentDuration, ...
                                     badGPSTimes, doRenormalize, modifyFilter, params.doOverlap, window1, ...
                                     window2, outputFileNamePrefix, displayResults, figureNumber, ...
                                     figureLegend, 0, [ params.outputFilePrefix '_resumeCRfMJ.mat' ], printFormats);
  fprintf('combineResults ptEstimate = %e\n', ptEstimate);
  fprintf('** Done combineResultsFromMultipleJobs\n');
  
  % Figures 9-17 (+ one new figure)
  % (there are only ten PanelPlots... "2" and "10" are missing)
  % params.doOverlap=1, DOFscalefactor = 1/(1+3/35) for 50% overlapping Hann
  StatisticalAnalysisofResults_v2(ptEstimate, concatCCStats, concatNaiveSigmas, ...
                                  dSigmaCut, params.segmentDuration, params.resampleRate1, ...
                                  window1, window2, figureLegend, [outputFileNamePrefix '_stats.dat'], ...
                                  1, 1/(1+3/35), outputFileNamePrefix, displayResults, applyBadGPSTimes, printFormats);
  fprintf('** Done StatisticalAnalysisofResults_v2\n');

  % Figure 8
  [tFFT, omega_t] = FFTofPtEstIntegrand(combinedPtEstInt, params.resampleRate1, ...
                                        outputFileNamePrefix, displayResults, figureNumber, ...
                                        figureLegend, false, printFormats);
  fprintf('** Done FFTofPtEstIntegrand\n');
  
  fprintf('** Done postProcessScriptFull\n');

return; % end postProcessScriptFull
