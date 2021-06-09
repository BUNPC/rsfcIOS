function varargout = rsfcGUI(varargin)
% RSFCGUI MATLAB code for rsfcGUI.fig
%      RSFCGUI, by itself, creates a new RSFCGUI or raises the existing
%      singleton*.
%
%      H = RSFCGUI returns the handle to a new RSFCGUI or the handle to
%      the existing singleton*.
%
%      RSFCGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RSFCGUI.M with the given input arguments.
%
%      RSFCGUI('Property','Value',...) creates a new RSFCGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before rsfcGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to rsfcGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help rsfcGUI

% Last Modified by GUIDE v2.5 12-Mar-2020 00:52:04

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @rsfcGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @rsfcGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before rsfcGUI is made visible.
function rsfcGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to rsfcGUI (see VARARGIN)

% Choose default command line output for rsfcGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes rsfcGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = rsfcGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbuttonTestMovie.
function pushbuttonTestMovie_Callback(hObject, eventdata, handles)
% hObject    handle to pushbuttonTestMovie (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global rsfc

% cross-correlate for different seeds


[nY,nX,nLam,nT] = size(rsfc.I);
brainIdx = find(rsfc.brain_mask == 1);
yAll = reshape(rsfc.I(:,:,1,:),[nY*nX,nT]);
y = yAll(brainIdx,:);
idx = find(isnan(mean(y,1)) == 0) 
y1 = y(:,idx);
rsfc.tRS1 = rsfc.tRS(idx);
nT1 = length(idx);

size(y)
size(y1)

y1 = interp1(rsfc.tRS1',y1',rsfc.tRS')';
yMean = mean(y1,1);
a = pinv(yMean*yMean')*yMean*y1';
ynew = y1'-yMean'*a;


h = 3;

Chb1 = zeros(nY*nX,nT);
size(Chb1)
size(ynew')
Chb1(brainIdx,:) = ynew';
Chb1 = reshape(Chb1, [nY, nX, nT]);
pROI = [60 40; 60 80; 74 34];
y2 = zeros(nT,size(pROI,1));

for ii=1:size(pROI,1)
    
    y2(:,ii) = squeeze(mean(mean(Chb1(pROI(ii,2)+[-h:h], pROI(ii,1)+[-h:h],:),1),2));
end



iLam = 1;


% bandpass filter

wn(2) = 0.08*2/rsfc.fs;
wn(1) = 0.009*2/rsfc.fs;
[fb,fa] = butter(5,wn,'bandpass');

y = reshape(Chb1,[nY*nX nT]);
% y = reshape(I(:,:,iLam,:),[nY*nX nT]);
y = y(brainIdx,:);

% lstInc = find(~isnan(mean(y,1)));
% y = y(:,lstInc);

y =  filtfilt(fb,fa,y')';


xLst = [20:2:75];
for iX = 1:length(xLst)
    pSeed = [xLst(iX) 80];
    
    ySeed = squeeze(mean(mean(Chb1(pSeed(2)+[-h:h],pSeed(1)+[-h:h],:),1),2));
%   ySeed = squeeze(mean(mean(I(pSeed(2)+[-h:h],pSeed(1)+[-h:h],iLam,lstInc),1),2));
    ySeed = filtfilt(fb,fa,ySeed);
    
    yCC = (y*ySeed) ./ (std(y,[],2)*std(ySeed)) / length(ySeed);
    
    Chb1CC = zeros(nY,nX);
    Chb1CC(brainIdx) = yCC;
    
    axes(handles.axes1)
    imagesc(Chb1CC,[-1 1])
    hold on
    plot(pSeed(1),pSeed(2),'o')
    hold off
    
    pause(0.5)
end
    


% --------------------------------------------------------------------
function menuLoad_Callback(hObject, eventdata, handles)
% hObject    handle to menuLoad (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% load your needed variables into rsfc.XXX rsfc.YYY etc

% clear  -global all

global  rsfc


[filename, pathname] = uigetfile('*_I.mat');

hwait = waitbar(0,'Loading...');
load([pathname filename]);
rsfc.I = I;
rsfc.Iunfilt = Iunfilt;
rsfc.I0 = I0;
rsfc.tRS = tRS;
rsfc.I0_unsampled = I0_unsampled;
rsfc.sampling_rate = size(rsfc.I0_unsampled,1)/size(rsfc.I0,1);

rsfc.atlas = imread('miceatlas.png');

waitbar(0.66,hwait)
 axes(handles.axes1)
  colormap('jet');
if (exist('rsfc_brainMask.mat','file') ~= 0) && (get(handles.radiobutton_brainmaskON,'Value') == get(handles.radiobutton_brainmaskON,'Max'))
  mask = load('rsfc_brainMask.mat');
  rsfc.brain_mask = mask.brain_mask;
  rsfc.Xi = mask.Xi;
  rsfc.Yi = mask.Yi;
  img = rsfc.I0(:,:,1).*uint16(rsfc.brain_mask);
  clims = [min(img(:)) max(img(:))];
  imagesc(img,clims);
else
  img = rsfc.I0(:,:,1);
  clims = [min(img(:)) max(img(:))];
  imagesc(img,clims);
end
   
axes(handles.axes4)
imagesc(rsfc.atlas);
close(hwait);
clear A I mask;




% --- Executes on mouse press over axes background.
function axes1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global rsfc

keyboard


% --- Executes on button press in pushbuttonSelectSeed.

 function pushbuttonSelectSeed_Callback(hObject, eventdata, handles)
% hObject    handle to pushbuttonSelectSeed (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global rsfc
global count;
global pts;
global str
global Chb1CC
global ct

ct = 0;
str = {};
cstring = 'rgbcmyk';
[nY,nX,nLam,nT] = size(rsfc.I);
nT = length(rsfc.tRS);
brainIdx = find(rsfc.brain_mask == 1);


hpf = str2num(get(handles.editHPF,'string'));
lpf = str2num(get(handles.editLPF,'string'));

wn(2) =lpf*2/rsfc.fs;
wn(1) = hpf*2/rsfc.fs;
if wn(1)>0 && wn(2)>0
    [fb,fa] = butter(5,wn,'bandpass');
elseif wn(2) > 0
    [fb,fa] = butter(5,wn(2),'low');
elseif wn(1) > 0
    [fb,fa] = butter(5,wn(2),'high');
end


pSeed = ginput(1);

% If the seed in selected in the baseline image. (Baseline image was unsampled and other images were downsampled.) 
if get(handles.togglebuttonbaselineimage,'value') == 1
    pSeed = pSeed/rsfc.sampling_rate;
end
    
pSeed = round(pSeed)    
if count == 0
    pts = pSeed;
else
    pts = [pts; pSeed];
end
    
y = rsfc.y;
h = str2double(get(handles.seedsize,'string'));

ySeed = squeeze(mean(mean(y(pSeed(2)+[-floor(h/2):floor(h/2)],pSeed(1)+[-floor(h/2):floor(h/2)],:),1),2));
%   ySeed = squeeze(mean(mean(I(pSeed(2)+[-h:h],pSeed(1)+[-h:h],iLam,lstInc),1),2));
if wn(1)>0 || wn(2)>0
   ySeed = filtfilt(fb,fa,ySeed);
end

ySeedunfilt = squeeze(mean(mean(rsfc.Iunfilt(pSeed(2)+[-floor(h/2):floor(h/2)],pSeed(1)+[-floor(h/2):floor(h/2)],:),1),2));
ySeedunGR = squeeze(mean(mean(rsfc.yunGR(pSeed(2)+[-floor(h/2):floor(h/2)],pSeed(1)+[-floor(h/2):floor(h/2)],:),1),2));

y = reshape(y,[nY*nX nT]);
y = y(brainIdx,:);
% yCC = (y*ySeed) ./ (std(y,[],2)*std(ySeed)) / length(ySeed);
yCC = corr(y.',ySeed);

Chb1CC = zeros(nY,nX);
Chb1CC(brainIdx) = yCC;
seedTC = mean(abs(yCC));

set(handles.text4,'String',num2str(seedTC))

axes(handles.axes1)

h = imagesc(Chb1CC,[-1 1]),colorbar;
colormap('jet');
set(h, 'ButtonDownFcn', {@mouseclick, handles});

for u = 1:size(pts,1)
    text(pts(u,1),pts(u,2),['O seed' num2str(u)], 'FontSize', 13);
end
for u = 1:length(rsfc.refX)
    text(rsfc.refX(u),rsfc.refY(u),'X', 'FontSize', 13);
end

% Dispaly time course of seed
axes(handles.axes2)
hold on

if get(handles.radiobutton_globalsignal,'Value') == get(handles.radiobutton_globalsignal,'Max')
    h = plot(rsfc.tRS,ySeedunfilt, cstring(mod(count,7)+1));
else
    h = plot(rsfc.tRS,ySeed, cstring(mod(count,7)+1));
end
for u = 1: count+1
    str{u} = ['seed' num2str(u)];
end
legend(str);
hold off
count = count+1;

% Display power spectrum of filtered and unfiltered data
L = length(ySeed);
NFFT = 2^nextpow2(L); % Next power of 2 from length of y
ySeedf = fft(ySeed,NFFT)/L;
f = rsfc.fs/2*linspace(0,1,NFFT/2+1);
axes(handles.axes3)
semilogx(f,2*abs(ySeedf(1:NFFT/2+1)),'b','LineWidth',2); 
hold on
xlim([0.005 1])
% f = log(f);
ySeedunfiltf = fft(ySeedunfilt,NFFT)/L;
semilogx(f,2*abs(ySeedunfiltf(1:NFFT/2+1)),'r','LineWidth',2); 
xlim([0.005 1])
hold on

ySeedunGRf = fft(ySeedunGR,NFFT)/L;
semilogx(f,2*abs(ySeedunGRf(1:NFFT/2+1)),'k','LineWidth',2); 
xlim([0.005 1])
hold off

filtpower = rsfc.df*sum(abs(ySeedf).^2);
unfiltpower = rsfc.df*sum(abs(ySeedunfiltf).^2);
filterunGRpower = rsfc.df*sum(abs(ySeedunGRf).^2);
set(handles.text_filteredpower,'String',num2str(filtpower));
set(handles.text_unfilteredpower,'String',num2str(unfiltpower));
set(handles.text21,'String',num2str(filterunGRpower));
% plotyy(f,2*abs(ySeedf(1:NFFT/2+1)),f,2*abs(ySeedunfilt(1:NFFT/2+1)));

% figure;
% colormap('gray');
% imagesc(rsfc.I0_unsampled',[0 800]);
% for u = 1:size(pts,1)
%     text(4*pts(u,2),4*pts(u,1),['O seed'], 'FontSize', 13);
% end
% 
if (get(handles.radiobutton_savetsps,'Value') == get(handles.radiobutton_savetsps,'Max'))
    figure(2);
    subplot(5,1,1)
    plot(rsfc.tRS,ySeedunfilt);
    xlabel('time');
    ylabel('magnitude')
    title('Unfiltered Signal');
    subplot(5,1,2)
    plot(rsfc.tRS,rsfc.yglobal);
    xlabel('time');
    ylabel('magnitude')
    title('Global Unfiltered Signal');
    subplot(5,1,3)
    plot(rsfc.tRS,ySeed)
    xlabel('time');
    ylabel('magnitude');
    title('Filtered Signal');
    subplot(5,1,4)
    semilogx(f,2*abs(ySeedf(1:NFFT/2+1)),'b','LineWidth',2); 
    hold on
    xlim([0.005 1]);
    semilogx(f,2*abs(ySeedunfiltf(1:NFFT/2+1)),'r','LineWidth',2); 
    title('Frequency Spectrum');
    xlabel('frequency');
    ylabel('power');
    subplot(5,1,5)
    semilogx(f,2*abs(rsfc.meanyallf(1:NFFT/2+1)),'b','LineWidth',2); 
    xlim([0.005 1]);
    title('Golabl Frequency Spectrum');
    xlabel('frequency');
    ylabel('power'); 
    
    figure(101);
    hold on;
    h = plot(rsfc.tRS,ySeed, cstring(mod(count,7)+1));
    xlabel('time(sec)');
    ylabel('magnitude');
    Ts = rsfc.tRS;
    save('plotdata.mat','ySeed','Ts');
    for u = 1: count+1
    str{u} = ['seed' num2str(u)];
    end
    legend(str);
    hold off;
    
    nameplot = inputdlg('Please enter the name to save the plot');
    ts = rsfc.tRS;
    save([nameplot{1} '.mat'],'ySeed','ts');

end

if (get(handles.radiobutton_SaveImage,'Value') == get(handles.radiobutton_SaveImage,'Max'))
    figure;
    colormap('jet');
    imagesc(Chb1CC,[-1 1]),colorbar;
    for u = 1:size(pts,1)
        text(pts(u,1),pts(u,2),['O seed' num2str(u)], 'FontSize', 18);
%         text(pts(u,2),pts(u,1),['O seed'], 'FontSize', 13);

    end
    name = inputdlg('Please enter the name to save the image');
    save([name{1} '.mat'],'Chb1CC');
end

set(handles.togglebuttonbaselineimage,'value',0);





% --- Executes on button press in pushbuttonPreProcess.
function pushbuttonPreProcess_Callback(hObject, eventdata, handles)
% hObject    handle to pushbuttonPreProcess (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global rsfc
global count 


count = 0;
hpf = str2num(get(handles.editHPF,'string'));
lpf = str2num(get(handles.editLPF,'string'));


hwait = waitbar(0,'Pre-processing...');

[nY,nX,nLam,nT] = size(rsfc.I);

% nY1 = size(rsfc.I0,1);
% sampling_rate = nY1/nY;
% brain_mask = rsfc.brain_mask(1:sampling_rate:end,1:sampling_rate:end);
% brainIdx = find(brain_mask == 1);

brainIdx = find(rsfc.brain_mask == 1);

% Selecting the data matrix        
yAll = reshape(rsfc.I(:,:,1,:),[nY*nX,nT]);
y1 = yAll(brainIdx,:);
tRS = rsfc.tRS;
% idx = find(isnan(mean(y,1)) == 0 & mean(y,1) < 220); 
% idx = find(isnan(mean(y,1)) == 0) ; 
% y1 = y(:,idx);
% tRS1 = rsfc.tRS(idx);
nT

% Removing data with artifacts
if isfield(rsfc,'excludeData') 
    disp('here');
    if (isempty(rsfc.excludeData) == 0)
        for u = 1:size(rsfc.excludeData,1)
            u
            idx1 = length(find(tRS<=rsfc.excludeData(u,1)));
            idx2 = length(find(tRS<=rsfc.excludeData(u,2)));
            if u == 1
                idxs = idx1:idx2;
            else
                idxs = [idxs idx1:idx2];
            end
        end
        y1(:,idxs) = [];
        tRS(idxs) = [];
        rsfc.I(:,:,:,idxs) = [];
        rsfc.excludeData = [];
        rsfc.Iunfilt(:,:,:,idxs) = [];
        axes(handles.axes2)
        cla reset
    end 
end

% nT1 = length(tRS1);
rsfc.tRS = tRS;
% y11 = y1;
% y1 = interp1(rsfc.tRS1',y1',rsfc.tRS')';
nT = length(tRS);
size(y1)
nY
nX
nT
rsfc.yunGR = zeros(nY*nX,nT);
rsfc.yunGR(brainIdx,:) = y1;
rsfc.yunGR = reshape(rsfc.yunGR,[nY nX nT]);
% size(y1)
% figure; plot(rsfc.tRS1,y11(5000,:)); figure; plot(rsfc.tRS,y1(5000,:));

% normalize data...
% Just testing this out... is it a good thing to do?
% we are trying it out so that pixels with large amplitude from spectular
% reflections don't dominate the mean
% y1 = y1./(mean(y1,2)*ones(1,size(y1,2)));


% subtract global mean
% if 0
% check matlab regression also
if get(handles.radiobutton_noGSR,'Value')
    ynew = y1';
elseif get(handles.radiobutton_GSR,'Value')
    yMean = mean(y1,1);
    a = pinv(yMean*yMean')*yMean*y1';
    size(a)
    ynew = y1'-yMean'*a;
else
    infarctIdx = find(rsfc.infarct_mask == 1);
    yAll = reshape(rsfc.I(:,:,1,:),[nY*nX,nT]);
    yinfarct = yAll(infarctIdx,:);
    yinfarctMean = mean(yinfarct,1);
%     a = pinv(yMean*yMean')*yMean*yinfarct';
%     yinfarctnew = yinfarct'-yMean'*a;
    
    noninfarctIdx = setdiff(brainIdx,infarctIdx);
    ynoninfarct = yAll(noninfarctIdx,:);
    ynoninfarctMean = mean(ynoninfarct,1);
%     a = pinv(yMean*yMean')*yMean*ynoninfarct';
%     ynoninfarctnew = ynoninfarct'-yMean'*a;

    yMean = [ynoninfarctMean; yinfarctMean];
    a = pinv(yMean*yMean')*yMean*y1';
    disp('size  of a is')
    size(a)
    ynew = y1'-yMean'*a;
    
%     yAll(infarctIdx,:) = yinfarctnew';
%     yAll(noninfarctIdx,:) = ynoninfarctnew';
%     
%     ynew = yAll(brainIdx,:);
%     ynew = ynew';
    
%     noninfarctIdx = setdiff(brainIdx,infarctIdx);
%     % Selecting the data matrix        
%     yAll = reshape(rsfc.I(:,:,1,:),[nY*nX,nT]);
%     yinfarct = yAll(infarctIdx,:);
%     ynoninfarct = yAll(noninfarctIdx,:);
%     yinfarctMean = mean(yinfarct,1);
%     ynoninfarctMean = mean(ynoninfarct,1);
%     yMean = yinfarctMean.*ynoninfarctMean;
%     a = pinv(yMean*yMean')*yMean*y1';
%     ynew = y1'-yMean'*a;
end
    size(ynew)
% else 
%     ynew = y1';
% end

% Chb1 = zeros(nY*nX,nT1);
% Chb1(brainIdx,:) = ynew';
% Chb1 = reshape(Chb1, [nY, nX, nT1]);

Chb1 = zeros(nY*nX,nT);
size(Chb1)
Chb1(brainIdx,:) = ynew';
Chb1 = reshape(Chb1, [nY, nX, nT]);



% bandpass filter
rsfc.fs = 1/(rsfc.tRS(2)-rsfc.tRS(1));
wn(2) = lpf*2/rsfc.fs;
wn(1) = hpf*2/rsfc.fs;
if wn(1)>0 && wn(2)>0
    [fb,fa] = butter(5,wn,'bandpass');
elseif wn(2) > 0
    [fb,fa] = butter(5,wn(2),'low');
elseif wn(1) > 0
    [fb,fa] = butter(5,wn(2),'high');
end
y = reshape(Chb1,[nY*nX nT]);
y = y(brainIdx,:);
if wn(1) > 0 || wn(2)>0
   y =  filtfilt(fb,fa,y')';
end
yy = zeros(nY*nX,nT);

yy(brainIdx,:) = y;
% yy = reshape(yy,[nY nX nT1]);
yy = reshape(yy,[nY nX nT]);

% nMin = 8;
% dt = 0.5;
% tRS = min(rsfc.tRS)+dt:dt:(nMin*60-dt);
% nT = length(tRS);
% yyy = zeros(nY,nX,nT);
% for iT = 1:nT
%     iT
%     lstIdx = find(rsfc.tRS>(tRS(iT)-dt) & rsfc.tRS<=(tRS(iT)+dt));
%     yyy(:,:,1,iT) = mean(yy(:,:,lstIdx),3);
% end
% rsfc.tRS = tRS;
rsfc.y = yy;


y = rsfc.y;
[nY,nX,nT] = size(y);
y = reshape(y,[nY*nX nT]);
axes(handles.axes1)
colormap(jet);
h = imagesc(rsfc.I0.*(rsfc.brain_mask));
set(h, 'ButtonDownFcn', {@mouseclick, handles});
brainIdx = find(rsfc.brain_mask == 1);
y = y(brainIdx,:);

hpf = str2num(get(handles.editHPF,'string'));
lpf = str2num(get(handles.editLPF,'string'));

wn(2) =lpf*2/rsfc.fs;
wn(1) = hpf*2/rsfc.fs;
if wn(1)>0 && wn(2)>0
    [fb,fa] = butter(5,wn,'bandpass');
elseif wn(2) > 0
    [fb,fa] = butter(5,wn(2),'low');
elseif wn(1) > 0
    [fb,fa] = butter(5,wn(2),'high');
end

I = rsfc.y;
t = rsfc.tRS;
save('Processeddata','I','t');

% create N*N correlation matrix (N = number of pixels in the brain)
% yCC = (y*y') ./ (std(y,[],2)*std(y,[],2)') / size(y,1);
yCC = corr(y.',y.');

load('atlas.mat');
rsfc.areaM = 8/(sqrt((xi(2)-xi(1))^2+(yi(2)-yi(1))^2));
yCCavg = mean(abs(yCC),1);
CI = mean(yCCavg);
set(handles.text_avgconnectivityglobal,'String',num2str(CI));
cutoff = str2num(get(handles.edit_cutoffCI,'string'));
idxs = find(yCCavg >= cutoff);
set(handles.text_globalArea,'String',num2str(length(idxs)*rsfc.areaM));
BOC = zeros(nY,nX);
BOC(brainIdx) = yCCavg;
axes(handles.axes5)
colormap(jet);
imagesc(BOC); colorbar;
rsfc.GCM = BOC;
rsfc.Sign_img = ((repmat((1:nY)',1,nX)-xi(1)).*(yi(2)-yi(1)))-((repmat(1:nX,nY,1)-yi(1)).*(xi(2)-xi(1)));
Oneside_idxs = find(rsfc.GCM >= cutoff & rsfc.Sign_img > 0 & rsfc.brain_mask == 1);
Otherside_idxs = find(rsfc.GCM >= cutoff & rsfc.Sign_img < 0 & rsfc.brain_mask == 1);
Oneside_idxs_CI = find( rsfc.Sign_img > 0 & rsfc.brain_mask == 1);
Otherside_idxs_CI = find( rsfc.Sign_img < 0 & rsfc.brain_mask == 1);
Oneside_CI = mean(abs(rsfc.GCM(Oneside_idxs_CI)));
Otherside_CI = mean(abs(rsfc.GCM(Otherside_idxs_CI)));
set(handles.text_ConnectivityIndex_1,'String',num2str(Oneside_CI ));
set(handles.text_ConnectivityIndex_2,'String',num2str(Otherside_CI));
set(handles.text_area_1,'String',num2str(length(Oneside_idxs)*rsfc.areaM));
set(handles.text_area_2,'String',num2str(length(Otherside_idxs)*rsfc.areaM));

% calculate gobal time series signal
rsfc.yglobal = squeeze(mean(mean(rsfc.Iunfilt,1),2));

% calculate global frequency signal
L = length(rsfc.yglobal);
NFFT = 2^nextpow2(L); % Next power of 2 from length of y
yall =  reshape(rsfc.Iunfilt,[nY*nX nT]);
yall = yall(brainIdx,:);
yallf = fft(yall',NFFT)/L;
rsfc.meanyallf = mean(abs(yallf),2);
rsfc.f = rsfc.fs/2*linspace(0,1,NFFT/2+1);
rsfc.df = (rsfc.f(2)-rsfc.f(1))/2;
globalpower = rsfc.df*sum(abs(rsfc.meanyallf).^2);
set(handles.text_globalpower,'String',num2str(globalpower));

% Left Right Spatial Correlation
axes(handles.axes6)
imagesc(rsfc.I0); colormap('gray');
if exist('BLpts.mat','file')
    load('BLpts.mat')
else
    h = msgbox(' Please select 2 points bregma and lambda in axes 6');
    uiwait(h);
    [y,x] = ginput(2);
    save('BLpts.mat','y','x');
end
a = y(2)-y(1);
b = x(1)-x(2);
c = (y(1)-y(2))*x(1)+(x(2)-x(1))*y(1);
rsfc.yLRC = zeros(nY,nX);
for u = 1:nY
    for v = 1:nX
        if rsfc.brain_mask(u,v) == 1
            xs = round(2*(b^2*u-a*b*v-a*c)/(a^2+b^2)-u);
            ys = round(2*(a^2*v-a*b*u-b*c)/(a^2+b^2)-v);
            if xs <1 
                xs = 1;
            end
            if xs > nY
                xs = nY;
            end
            if ys <1 
                ys = 1;
            end
            if ys > nX
                ys = nX;
            end
            
            h = str2double(get(handles.seedsize,'string'));
            x1_min = min(max(xs-floor(h/2),1),nY);
            x1_max = min(max(xs+floor(h/2),1),nY);
            y1_min = min(max(ys-floor(h/2),1),nY);
            y1_max = min(max(ys+floor(h/2),1),nY);
            x2_min = min(max(u-floor(h/2),1),nY);
            x2_max = min(max(u+floor(h/2),1),nY);
            y2_min = min(max(v-floor(h/2),1),nY);
            y2_max = min(max(v+floor(h/2),1),nY);
%             CR = corr(squeeze(rsfc.y(xs,ys,:)),squeeze(rsfc.y(u,v,:)));
            CR = corr(squeeze(mean(mean(rsfc.y(x1_min:x1_max,y1_min:y1_max,:),1),2)),squeeze(mean(mean(rsfc.y(x2_min:x2_max,y2_min:y2_max,:),1),2)));
            rsfc.yLRC(xs,ys) = CR;
            rsfc.yLRC(u,v) = CR;
        end
    end 
end
imagesc(rsfc.yLRC,[-1 1]); colormap('jet'); colorbar;
LRC = rsfc.yLRC;
save('LRC.mat','LRC');
close(hwait);



% --- Executes on button press in togglebuttonbaselineimage.
function togglebuttonbaselineimage_Callback(hObject, eventdata, handles)
% hObject    handle to togglebuttonbaselineimage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of togglebuttonbaselineimage

global rsfc
global pts
global Chb1CC

axes(handles.axes1);

if get(hObject,'Value') == 1
   colormap('gray');
%  img = rsfc.I0(:,:,1).*rsfc.brain_mask;
   img = rsfc.I0_unsampled(:,:,1); 
   clims = [min(img(:)) max(img(:))];
   imagesc(img,[rsfc.MinS rsfc.MaxS]); colorbar;
   hold on
   for u = 1:size(pts,1)
       text(rsfc.sampling_rate*pts(u,1),rsfc.sampling_rate*pts(u,2),['O seed' num2str(u)], 'FontSize', 13);
   end
   for u = 1:length(rsfc.refX)
    text(rsfc.sampling_rate*rsfc.refX(u),rsfc.sampling_rate*rsfc.refY(u),'X', 'FontSize', 13);
   end
   hold off
  
else
   colormap('jet');
   h = imagesc(Chb1CC,[-1 1]); colorbar;
   hold on
   for u = 1:size(pts,1)
       text(pts(u,1),pts(u,2),['O seed' num2str(u)], 'FontSize', 13);
   end
   for u = 1:length(rsfc.refX)
    text(rsfc.refX(u),rsfc.refY(u),'X', 'FontSize', 13);
   end
   hold off
   set(h, 'ButtonDownFcn', {@mouseclick, handles});
end




% --- Executes on button press in togglebutton2.
function togglebutton2_Callback(hObject, eventdata, handles)
% hObject    handle to togglebutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of togglebutton2
global count
global pts
global rsfc
count = 0;
pts = [0 0];
axes(handles.axes2)
cla reset
axes(handles.axes3);
cla reset
axes(handles.axes1);
cla reset
colormap('jet');
% img = rsfc.I0(:,:,1).*rsfc.brain_mask;
img = rsfc.I0(:,:,1);
clims = [min(img(:)) max(img(:))];
h = imagesc(img,clims);
for u = 1:length(rsfc.refX)
    text(rsfc.refX(u),rsfc.refY(u),'X', 'FontSize', 13);
end
set(h, 'ButtonDownFcn', {@mouseclick, handles});
set(handles.togglebuttonbaselineimage,'value',0);


% --- Executes on button press in pushbuttonbrainmask.
function pushbuttonbrainmask_Callback(hObject, eventdata, handles)
% hObject    handle to pushbuttonbrainmask (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global rsfc
final_brain_mask = false(size(rsfc.I0));
while(1)
  axes(handles.axes1)
  MinMinS = min(rsfc.I0(:)); MaxMaxS = max(rsfc.I0(:));
  while(1)
    imagesc(rsfc.I0(:,:,1),[ MinMinS MaxMaxS]); colorbar;
    colormap('gray');
    button = questdlg('Are you happy with the image contrast?','Repeat selection or not...','No');
    if strcmp(button,'No')
        answer= inputdlg({'Enter minimum intensity level of survey scan image'},'Enter minimum intensity level:',1,{num2str(MinMinS)});  %    {'0'});
        answer1 = cell2struct(answer, 'number', 1);
        MinMinS = str2num(answer1.number);
        answer= inputdlg({'Enter maximum intensity level of survey scan image'},'Enter maximum intensity level:',1,{num2str(MaxMaxS)});  %    {'0'});
        answer1 = cell2struct(answer, 'number', 1);
        MaxMaxS = str2num(answer1.number);
    else 
        break;
    end
  end
  rsfc.MaxS = MaxMaxS;
  rsfc.MinS = MinMinS;

  h = msgbox(' Please select the brain region');
  uiwait(h)
%   axes(handles.axes6)
%   imagesc(rsfc.yLRC,[-1 1]); 
  [brain_mask,Xi,Yi] = roipoly;
  hold on;
  plot(Xi,Yi,'color','k');
  hold off;
  button = questdlg('Are you statisfied with ROI?');
  if strcmp(button, 'Yes')
      final_brain_mask(brain_mask==1) = 1;
      button = questdlg('Do you want to select another ROI?');
      if ~strcmp(button, 'Yes')
        break;
      end
  end
end
  if (get(handles.radiobutton4,'Value') == get(handles.radiobutton4,'Max'))
    figure; imshow(rsfc.I0,[MinMinS MaxMaxS]); colormap('gray');
  end
brain_mask = final_brain_mask;
save('rsfc_brainMask.mat','brain_mask','Xi','Yi');
rsfc.brain_mask = final_brain_mask;
rsfc.Xi = Xi;
rsfc.Yi = Yi;


% --- Executes on button press in pushbutton_slecttime.
function pushbutton_slecttime_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_slecttime (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global rsfc
global tidx
global h
global ct
if ct > 0
    delete(h);
end
ct = 1;
axes(handles.axes2)
[X,~] =ginput(1);
tidx = length(find(rsfc.tRS < X))+1;
% tp = rsfc.tRS(idx);
axes(handles.axes1)
colormap('jet');
if (get(handles.radiobutton_brainmaskON,'Value') == get(handles.radiobutton_brainmaskON,'Max'))
    img = rsfc.I(:,:,1,tidx).*rsfc.brain_mask;
    clims = [min(img(:)) max(img(:))];
    imagesc(img,clims);
else
    img = rsfc.I0(:,:,1);
    clims = [min(img(:)) max(img(:))];
    imagesc(img,clims);
end
axes(handles.axes2)
hold on
h = text(rsfc.tRS(tidx),0,'X','FontSize',15);
hold off


% --- Executes on button press in pushbutton_forward.
function pushbutton_forward_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_forward (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global rsfc
global tidx
global h
delete(h)
tidx = tidx+1;
axes(handles.axes1)
colormap('jet');
if (get(handles.radiobutton_brainmaskON,'Value') == get(handles.radiobutton_brainmaskON,'Max'))
    img = rsfc.I(:,:,1,tidx).*rsfc.brain_mask;
    clims = [min(img(:)) max(img(:))];
    imagesc(img,clims);
else
    img = rsfc.I(:,:,1,tidx);
    colormap('gray');
    clims = [min(img(:)) max(img(:))];
    imagesc(img,clims);
end
axes(handles.axes2)
hold on
h = text(rsfc.tRS(tidx),0,'X','FontSize',13);
hold off


% --- Executes on button press in pushbutton_backward.
function pushbutton_backward_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_backward (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global rsfc
global tidx
global h
delete(h)

tidx = tidx-1;
axes(handles.axes1)
colormap('jet');
if (get(handles.radiobutton_brainmaskON,'Value') == get(handles.radiobutton_brainmaskON,'Max'))
    img = rsfc.I(:,:,1,tidx).*rsfc.brain_mask;
    clims = [min(img(:)) max(img(:))];
    imagesc(img,clims);
else
    img = rsfc.I(:,:,1,tidx);
    colormap('gray');
    clims = [min(img(:)) max(img(:))];
    imagesc(img,clims);
end
axes(handles.axes2)
hold on
h = text(rsfc.tRS(tidx),0,'X','FontSize',13);
hold off



function editHPF_Callback(hObject, eventdata, handles)
% hObject    handle to editHPF (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editHPF as text
%        str2double(get(hObject,'String')) returns contents of editHPF as a double



function editLPF_Callback(hObject, eventdata, handles)
% hObject    handle to editLPF (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editLPF as text
%        str2double(get(hObject,'String')) returns contents of editLPF as a double


% --- Executes on button press in pushbutton.
function pushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global rsfc
global obj_count
k = waitforbuttonpress;
if k==0
     pt1 = get(gca,'CurrentPoint');  % button down detected
     finalRect = rbbox;                   % return figure units
     pt2 = get(gca,'CurrentPoint');
     
     %Checking boundary conditions
     if pt1(1,1) < rsfc.tRS(1)
         pt1(1,1) = rsfc.tRS(1);
     end
     if pt1(1,1) > rsfc.tRS(end)
         pt1(1,1) = rsfc.tRS(end);
     end
     if pt2(1,1) < rsfc.tRS(1)
         pt2(1,1) = rsfc.tRS(1);
     end
     if pt2(1,1) > rsfc.tRS(end)
         pt2(1,1) = rsfc.tRS(end);
     end
     
     %Arrange pts in increasing order 
     if pt1(1,1) > pt2(1,1)
         temp = pt1(1,1);
         pt1(1,1) = pt2(1,1);
         pt2(1,1) = temp;
     end
     
     
      if isfield(rsfc,'excludeData')
         rsfc.excludeData(end+1,:) = [pt1(1,1) pt2(1,1)];
      else
          rsfc.excludeData(1,:) = [pt1(1,1) pt2(1,1)];
      end
     axes(handles.axes2)
     yy = ylim();
     hp = patch([pt1(1,1) pt2(1,1) pt2(1,1) pt1(1,1)],[yy(1) yy(1) yy(2) yy(2)],'m','facealpha',0.3,'edgecolor','k');
     idx = size(rsfc.excludeData,1);
     rsfc.excludeData_handle(idx) = hp;
     obj_count(idx) = idx;
     set(hp,'ButtonDownFcn', sprintf('rsfc_excludeData_PatchCallback(%d)',idx) );
end  


% --- Executes on button press in radiobutton_brainmaskON.
function radiobutton_brainmaskON_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton_brainmaskON (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton_brainmaskON


% --- Executes on button press in pushbutton_iterrativeparcellation.
function pushbutton_iterrativeparcellation_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_iterrativeparcellation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global rsfc

y = rsfc.y;
[nY,nX,nT] = size(y);
y = reshape(y,[nY*nX nT]);
brainIdx = find(rsfc.brain_mask == 1);
y = y(brainIdx,:);

hpf = str2num(get(handles.editHPF,'string'));
lpf = str2num(get(handles.editLPF,'string'));

wn(2) =lpf*2/rsfc.fs;
wn(1) = hpf*2/rsfc.fs;
if wn(1)>0 && wn(2)>0
    [fb,fa] = butter(5,wn,'bandpass');
elseif wn(2) > 0
    [fb,fa] = butter(5,wn(2),'low');
elseif wn(1) > 0
    [fb,fa] = butter(5,wn(2),'high');
end


% create N*N correlation matrix (N = number of pixels in the brain)
% yCC = (y*y') ./ (std(y,[],2)*std(y,[],2)') / size(y,1);
yCC = corr(y.',y.');

% yCCavg = mean(abs(yCC),1);
% BOC = zeros(nY,nX);
% BOC(brainIdx) = yCCavg;
% figure; imagesc(BOC); colorbar;

% Singular Value Decomposition
% [U,~,V] = svd(yCC);
axes(handles.axes1)
while(1)
    % intial assumption of parcels based on first ten modes of connectivity matrix
    yatlas = zeros(1,length(brainIdx));
    l = length(yatlas);
    button = questdlg('Would you like to calculate another parcellation map ?');
    if strcmp(button,'Yes')
        button2 = questdlg('Would you like to open saved seeds for parcellation');
        if strcmp(button2,'Yes')
            [filename,pathname] = uigetfile;
            load([pathname filename]); 
            load('atlas.mat');
            ya = [328 328]; xa = [129 459];
            fixedPoints = [[xi(1) yi(1)]; [xi(2) yi(2)]];
            movingPoints = [[xa(1) ya(1)]; [xa(2) ya(2)]];
            tform = fitgeotrans(movingPoints,fixedPoints,'nonreflectivesimilarity');
            [xp,yp] = transformPointsForward(tform, xp, yp);
            yp = round(yp);
            xp = round(xp);
        else
            answer = char(inputdlg('Please enter number of parcellations'));
            psize = str2num(answer);
    %         for u = 1:l
    %             [~,id] = max(abs(U(u,1:psize)));
    %             yatlas(u) = U(u,id)*id/abs(U(u,id));
    %         end
    %         yintial_atlas = yatlas;
    %         idx = find(yatlas < 0);
    %         yatlas(idx) = yatlas(idx) + (2*psize+1);

            % For seed based parcellation

            imagesc(rsfc.I0);
            colormap('gray');
            h = msgbox('Please select the seeds for parcellation');
            uiwait(h);
            [yp,xp] = ginput(psize);
            yp = round(yp);
            xp = round(xp);
        end
        p_timetraces = zeros(length(xp),nT);
        for u  = 1:size(p_timetraces,1)
            p_timetraces(u,:) = rsfc.y(xp(u),yp(u),:);
        end
        yR = corr(y.',p_timetraces.');
        for w = 1:l
            [~,id] = max(yR(w,:));
            yatlas(w) = id;
        end
        brain_atlas = zeros(nY,nX);
        brain_atlas(brainIdx) = yatlas;
        figure; imagesc(brain_atlas); colorbar;
  
        
        % iterative parcellation (clustering algorithm)
%         p_timetraces = zeros(2*psize,nT);
        count = 0;
        u = 0;
        
        while(1)
            u = u+1;
        %     for v = 1:10
        %         idx1 = find(yatlas == -v);
        %         idx2 = find(yatlas == v);
        %         p_timetraces(v,:) = mean(y(idx1,:),1);
        %         p_timetraces(v+10,:) = mean(y(idx2,:),1);
        %     end
             for v = 1:size(p_timetraces,1)
        %         idx1 = find(yatlas == -v);
        %         idx2 = find(yatlas == v);
                  idx = find(yatlas == v);
                  p_timetraces(v,:) = mean(y(idx,:),1);
        %         p_timetraces(v+10,:) = mean(y(idx2,:),1);
                  if idx < 50
                      if  count == 0
                          lst = v;
                          count = count+1;
                      else
                          lst = [lst idx];
                          count = count+1;
                      end
                  end
             end

            if count > 0
                p_timetraces(lst,:) = [];
                lst = [];
                count = 0;

            end

            if wn(1)>0 || wn(2)>0
                p_timetraces = filtfilt(fb,fa,p_timetraces')';
            end

        %     yR = (y*p_timetraces')./ (std(y,[],2)*std(p_timetraces,[],2)') / size(y,1);
              yR = corr(y.',p_timetraces.');
            for w = 1:l
                [~,id] = max(yR(w,:));
                yatlas(w) = id;
            end
            
            if u > 1
                if isequal(yatlas,yatlasCond)
                    break;
                end
            end
            yatlasCond = yatlas;

        %     idx1 = find(yatlas < 11);
        %     yatlas(idx1) = -1*yatlas(idx1);
        %     idx2 = find(yatlas > 10);
        %     yatlas(idx2) = yatlas(idx2)-10;
        end
        no_of_iterations = u


        brain_atlas = zeros(nY,nX);
        brain_atlas(brainIdx) = yatlas;
        figure; imagesc(brain_atlas); colorbar;
        save('brain_atlas.mat','brain_atlas');
     

%         pR = corr(p_timetraces.',p_timetraces.');
%         C = linkage(pR);
%         figure; dendrogram(C);
    else
        break;
    end
end






        

     
    

% U = U(:,1:10);
% Rmatrix = (y*U')./(std(y,[],2)'*std(U,[],2))/ size(y,1);
% for u = 1:l
%     [min,id] = unique(Rmatrix(1:10,u));
%     yatlas(u) = min*id/abs(min);
% end


    



function seedsize_Callback(hObject, eventdata, handles)
% hObject    handle to seedsize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of seedsize as text
%        str2double(get(hObject,'String')) returns contents of seedsize as a double


% --- Executes during object creation, after setting all properties.
function seedsize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to seedsize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_avgfreqspectrum.
function pushbutton_avgfreqspectrum_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_avgfreqspectrum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global rsfc

y = rsfc.y;
[nY,nX,nT] = size(y);
y = reshape(y,[nY*nX nT]);
brainIdx = find(rsfc.brain_mask == 1);
y = y(brainIdx,:);

hpf = str2num(get(handles.editHPF,'string'));
lpf = str2num(get(handles.editLPF,'string'));

wn(2) =lpf*2/rsfc.fs;
wn(1) = hpf*2/rsfc.fs;
if wn(1)>0 && wn(2)>0
    [fb,fa] = butter(5,wn,'bandpass');
elseif wn(2) > 0
    [fb,fa] = butter(5,wn(2),'low');
elseif wn(1) > 0
    [fb,fa] = butter(5,wn(2),'high');
end


% create N*N correlation matrix (N = number of pixels in the brain)
% yCC = (y*y') ./ (std(y,[],2)*std(y,[],2)') / size(y,1);
yCC = corr(y.',y.');

yCCavg = mean(abs(yCC),1);
BOC = zeros(nY,nX);
BOC(brainIdx) = yCCavg;
figure; imagesc(BOC,[0 0.5]); colormap('jet'); colorbar;




% y = rsfc.y;
% fs = 11;
% [nY,nX,nT] = size(y);
% y = reshape(y,[nY*nX nT]);
% brainIdx = find(rsfc.brain_mask == 1);
% y = y(brainIdx,:);
% L = size(y,1);
% NFFT = 2^nextpow2(L); % Next power of 2 from length of y
% yf = fft(y',NFFT)'/L;
% f = fs/2*linspace(0,1,NFFT/2+1);
% yfavg = mean(yf,1);
% figure;
% semilogx(f,2*abs(yfavg(1:NFFT/2+1)),'b','LineWidth',2); 


% --- Executes on button press in radiobutton_SaveImage.
function radiobutton_SaveImage_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton_SaveImage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton_SaveImage


% --- Executes on button press in pushbutton11.
function pushbutton11_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global rsfc
y = rsfc.y;

[AVIfileOut, AVIpathOut] = uiputfile( '*.avi','Create AVI movie name...');
AVIfileName = [AVIpathOut AVIfileOut]; 
writerObj = VideoWriter(AVIfileName,'Uncompressed AVI');
writerObj.FrameRate = 40;
open(writerObj);
l = size(y,3);
cmap = colormap(jet(256));
% cmap = cmap(1:51,:);
ymin = min(y(:));
ymax = max(y(:));
y = (y-ymin)/(ymax-ymin);
MinMinS = min(y(:));
MaxMaxS = max(y(:));
while(1)
    figure(1); imagesc(y(:,:,1),[MinMinS MaxMaxS]);
    colormap(jet);
    colorbar;
    button = questdlg('Are you happy with the image contrast?','Repeat selection or not...','No');
    if strcmp(button,'No')
        answer= inputdlg({'Enter minimum intensity level of survey scan image'},'Enter minimum intensity level:',1,{num2str(MinMinS)});  %    {'0'});
        answer1 = cell2struct(answer, 'number', 1);
        MinMinS = str2num(answer1.number);
        answer= inputdlg({'Enter maximum intensity level of survey scan image'},'Enter maximum intensity level:',1,{num2str(MaxMaxS)});  %    {'0'});
        answer1 = cell2struct(answer, 'number', 1);
        MaxMaxS = str2num(answer1.number);
    else 
        break;
    end
end
for u = 1:l
    figure(111)
    imagesc(squeeze(y(:,:,u)),[MinMinS MaxMaxS]);
    text(10,10,[num2str(int16(rsfc.tRS(u))) 'Secs']);
    colormap(jet);
    F = getframe(gcf);
    writeVideo(writerObj,F);
    %mov = addframe(mov,F);
end;

%mov = close(mov);
close(writerObj);

    
    
    
    
    
    


% --- Executes on button press in pushbutton_registeratlas.
function pushbutton_registeratlas_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_registeratlas (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global rsfc


h = msgbox('Please select two points in the image');
uiwait(h);

% reading the points Bregma and Lambda from image.
MinMinS = min(rsfc.I0(:)); MaxMaxS = max(rsfc.I0(:));
axes(handles.axes1)
while(1)
    imagesc(rsfc.I0(:,:,1),[ MinMinS MaxMaxS]); colorbar;
    colormap('gray');
    button = questdlg('Are you happy with the image contrast?','Repeat selection or not...','No');
    if strcmp(button,'No')
        answer= inputdlg({'Enter minimum intensity level of survey scan image'},'Enter minimum intensity level:',1,{num2str(MinMinS)});  %    {'0'});
        answer1 = cell2struct(answer, 'number', 1);
        MinMinS = str2num(answer1.number);
        answer= inputdlg({'Enter maximum intensity level of survey scan image'},'Enter maximum intensity level:',1,{num2str(MaxMaxS)});  %    {'0'});
        answer1 = cell2struct(answer, 'number', 1);
        MaxMaxS = str2num(answer1.number);
    else 
        break;
    end
end
rsfc.MaxS = MaxMaxS;
rsfc.MinS = MinMinS;
[y,x] = ginput(2)
yi = y;
xi = x;
save('atlas.mat','yi','xi');

% Bregma and Lambda locations from atlas.
ya = [328 328]; xa = [129 459];

% Since trying to register atlas to the data, atlas points are moving
% points and image points are fixed points
fixedPoints = [[x(1) y(1)]; [x(2) y(2)]];
movingPoints = [[xa(1) ya(1)]; [xa(2) ya(2)]];

[m1,n1] = size(rsfc.I0)
[m2,n2,~] = size(rsfc.atlas);
tform = fitgeotrans(movingPoints,fixedPoints,'nonreflectivesimilarity');
count = 0;
% img3(:,:,1) = imwarp(rsfc.atlas(:,:,1),tform);
% img3(:,:,2) = imwarp(rsfc.atlas(:,:,2),tform);
% img3(:,:,3) = imwarp(rsfc.atlas(:,:,3),tform);
u = repmat((1:m2)',1,n2);
v = repmat(1:n2,m2,1);
[ut,vt] = transformPointsForward(tform, u(:), v(:));
img3 = rsfc.atlas(1:m1,1:n1,:)*0;
L = length(u(:));
for xx = 1:L
    if ut(xx) < 1 || ut(xx) > m1 || vt(xx) < 1 || vt(xx) > n1 
    else
        img3(round(ut(xx)),round(vt(xx)),:) = rsfc.atlas(u(xx),v(xx),:);
    end
end

% for  xx = 1:
% img3(round(ut),round(vt),:) = rsfc.atlas(u,v,:);
%  figure;
%  imagesc(img3)
%  for u = 1:m2
% %      u
%      for v = 1:n2
%          [tempx,tempy] = transformPointsForward(t,v,u);
% %          [tempx,tempy] = tforminv(t,u,v);
%          if tempx < 1 || tempx > n1 || tempy < 1 || tempy > m1
%          else
%              count = count+1;
%              img3(round(tempy),round(tempx),:) = rsfc.atlas(u,v,:);
% %              img3(u,v,:) = img1(round(tempy),round(tempx),:);
%          end
%      end
%  end
%  count
%  u
%  v
 axes(handles.axes4)
 imagesc(img3)

%  axis([0,m1,0,n1]);
 
 


% --- Executes during object creation, after setting all properties.
function text4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in pushbutton_selectROI.
function pushbutton_selectROI_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_selectROI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global rsfc

GCM = rsfc.GCM;
while(1)
    axes(handles.axes5)
    imagesc(rsfc.GCM,[0 max(GCM(:))]); colorbar;
    for u = 1:length(rsfc.refX)
    text(rsfc.refX(u),rsfc.refY(u),'X', 'FontSize', 13);
    end
  [mask,Xi,Yi] = roipoly;
  axes(handles.axes5)
  hold on;
  plot(Xi,Yi,'color','k');
  button = questdlg('Are you statisfied with ROI');
  if strcmp(button, 'Yes');
      break;
  end
end
cutoff = str2num(get(handles.edit_cutoffCI,'string'));
idxs1 = find(mask==1);
idxs2 = find(mask==1 & rsfc.GCM >= cutoff);
set(handles.text_areaofROI,'String',num2str(length(idxs2)));
if(isempty(idxs1)==0)
    set(handles.text_avgconnectivity,'String',num2str(mean(GCM(idxs1))));
else
    set(handles.text_avgconnectivity,'String','0');
end

axes(handles.axes5)
    imagesc(rsfc.GCM,[cutoff max(GCM(:))]); colorbar;
     plot(Xi,Yi,'color','k');
     for u = 1:length(rsfc.refX)
        text(rsfc.refX(u),rsfc.refY(u),'X', 'FontSize', 13);
     end
     
rsfc.Smask = mask;
rsfc.SXi = Xi;
rsfc.SYi = Yi;

function edit_cutoffCI_Callback(hObject, eventdata, handles)
% hObject    handle to edit_cutoffCI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_cutoffCI as text
%        str2double(get(hObject,'String')) returns contents of edit_cutoffCI as a double

global rsfc

cutoff = str2num(get(handles.edit_cutoffCI,'string'));
idxs = find(rsfc.brain_mask==1 & rsfc.GCM >= cutoff);
set(handles.text_globalArea,'String',num2str(length(idxs)*rsfc.areaM));
Oneside_idxs = find(rsfc.GCM >= cutoff & rsfc.Sign_img > 0 & rsfc.brain_mask == 1);
Otherside_idxs = find(rsfc.GCM >= cutoff & rsfc.Sign_img < 0 & rsfc.brain_mask == 1);
% Oneside_CI = mean(abs(rsfc.GCM(Oneside_idxs)));
% Otherside_CI = mean(abs(rsfc.GCM(Otherside_idxs)));
% set(handles.text_ConnectivityIndex_1,'String',num2str(Oneside_CI ));
% set(handles.text_ConnectivityIndex_2,'String',num2str(Otherside_CI));
set(handles.text_area_1,'String',num2str(length(Oneside_idxs)*rsfc.areaM));
set(handles.text_area_2,'String',num2str(length(Otherside_idxs)*rsfc.areaM));
if isfield(rsfc,'Smask') 
    cutoff = str2num(get(handles.edit_cutoffCI,'string'));
    idxs2 = find(rsfc.Smask==1 & rsfc.GCM >= cutoff);
    set(handles.text_areaofROI,'String',num2str(length(idxs2)*rsfc.areaM));
end
axes(handles.axes5)
    if  cutoff >= max(rsfc.GCM(:))
        maxI = cutoff+0.1;
    else 
        maxI = max(rsfc.GCM(:));
    end
    imagesc(rsfc.GCM,[cutoff maxI]); colorbar;
    if isfield(rsfc,'Smask') 
        plot(rsfc.SXi,rsfc.SYi,'color','k');
    end
    for u = 1:length(rsfc.refX)
        text(rsfc.refX(u),rsfc.refY(u),'X', 'FontSize', 13);
    end




% --- Executes during object creation, after setting all properties.
function edit_cutoffCI_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_cutoffCI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_minI_Callback(hObject, eventdata, handles)
% hObject    handle to edit_minI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_minI as text
%        str2double(get(hObject,'String')) returns contents of edit_minI as a double


% --- Executes during object creation, after setting all properties.
function edit_minI_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_minI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_MaxI_Callback(hObject, eventdata, handles)
% hObject    handle to edit_MaxI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_MaxI as text
%        str2double(get(hObject,'String')) returns contents of edit_MaxI as a double


% --- Executes during object creation, after setting all properties.
function edit_MaxI_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_MaxI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function mouseclick(hObject, eventdata, handles)
    
 
global rsfc
parent = (get(hObject, 'Parent')) ;
pts = round(get(parent, 'CurrentPoint'));
% 
cutoff = str2num(get(handles.edit_cutoffCI,'string'));
if  cutoff >= max(rsfc.GCM(:))
    maxI = cutoff+0.1;
else 
    maxI = max(rsfc.GCM(:));
end
axes(handles.axes5)
imagesc(rsfc.GCM,[cutoff maxI]); colorbar;
y = [pts(1,2) pts(1,2)];
x = xlim();
line(x,y,'LineWidth',2,'color','r');
y = ylim();
x = [pts(1,1), pts(1,1)];
line(x,y,'LineWidth',2,'color','r');



    


% --- Executes on button press in pushbutton_SelectReferencePoints.
function pushbutton_SelectReferencePoints_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_SelectReferencePoints (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global rsfc

axes(handles.axes1)
imagesc(rsfc.I0);
[X,Y] = ginput(3);
rsfc.refX = X;
rsfc.refY = Y;
for u = 1:length(X)
    text(X(u),Y(u),'X', 'FontSize', 13);
end

% axes(handles.axes5)
% imagesc(rsfc.GCM);
% for u = 1:length(rsfc.refX)
%     text(rsfc.refX(u),rsfc.refY(u),'X', 'FontSize', 13);
% end


% --- Executes on button press in radiobutton_globalsignal.
function radiobutton_globalsignal_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton_globalsignal (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton_globalsignal


% --- Executes on button press in radiobutton4.
function radiobutton4_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton4


% --- Executes on button press in radiobutton_savetsps.
function radiobutton_savetsps_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton_savetsps (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton_savetsps


% --- Executes on button press in pushbutton_atlasseeds.
function pushbutton_atlasseeds_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_atlasseeds (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global rsfc


% --- Executes on button press in radiobutton_globaltimesignal.
function radiobutton_globaltimesignal_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton_globaltimesignal (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton_globaltimesignal
global rsfc
if (get(handles.radiobutton_globaltimesignal,'Value') == get(handles.radiobutton_brainmaskON,'Max'))
    axes(handles.axes2)
    hold on
    plot(rsfc.tRS,rsfc.yglobal);
end
% --- Executes on button press in radiobutton_globalfreqsignal.
function radiobutton_globalfreqsignal_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton_globalfreqsignal (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton_globalfreqsignal

global rsfc
if (get(handles.radiobutton_globalfreqsignal,'Value') == get(handles.radiobutton_brainmaskON,'Max'))
    L = length(rsfc.yglobal);
    NFFT = 2^nextpow2(L); 
    axes(handles.axes3)
    hold on
    semilogx(rsfc.f,2*abs(rsfc.meanyallf(1:NFFT/2+1)),'k','LineWidth',2); 
    xlim([0.005 1])
end


% --- Executes on button press in pushbutton_autoanalysis.
function pushbutton_autoanalysis_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_autoanalysis (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global rsfc

y = rsfc.y;
[nY, nX, nT] = size(y);
brainIdx = find(rsfc.brain_mask == 1);
h1 = msgbox('Please select the seeds file');
uiwait(h1);
[filename,pathname] = uigetfile;
h2 = msgbox('please select the folder for output files');
uiwait(h2);
pathnameout = uigetdir;
load([pathname filename]);
load('atlas.mat');
ya = [328 328]; xa = [129 459];
fixedPoints = [[xi(1) yi(1)]; [xi(2) yi(2)]];
movingPoints = [[xa(1) ya(1)]; [xa(2) ya(2)]];
tform = fitgeotrans(movingPoints,fixedPoints,'nonreflectivesimilarity');
[xp,yp] = transformPointsForward(tform, xp, yp);
yp = round(yp);
xp = round(xp);
h = str2num(get(handles.seedsize,'string'));%Seed dimension will be h*2+1 x h*2+1
seedTC = zeros(length(xp),1);
filtandGRpower = zeros(length(xp),1);
unfiltpower= zeros(length(xp),1);
filteredpower= zeros(length(xp),1);
y1 = reshape(y,[nY*nX nT]);
y1 = y1(brainIdx,:);
for u=1:length(xp)
    ySeed = squeeze(mean(mean(y(xp(u)+[-floor(h/2):floor(h/2)],yp(u)+[-floor(h/2):floor(h/2)],:),1),2));
    ySeedunfilt = squeeze(mean(mean(rsfc.Iunfilt(xp(u)+[-floor(h/2):floor(h/2)],yp(u)+[-floor(h/2):floor(h/2)],:),1),2));
    ySeedunGR = squeeze(mean(mean(rsfc.yunGR(xp(u)+[-floor(h/2):floor(h/2)],yp(u)+[-floor(h/2):floor(h/2)],:),1),2));
    % yCC = (y*ySeed) ./ (std(y,[],2)*std(ySeed)) / length(ySeed);
    yCC = corr(y1.',ySeed);
    Chb1CC = zeros(nY,nX);
    Chb1CC(brainIdx) = yCC;
    f1 = figure(101); imagesc(Chb1CC,[-1 1]); colormap('jet'); text(yp(u),xp(u),'X');
    saveas(f1, [pathnameout '/Seed' num2str(u) '.tif']);
    close(f1);
   
    
    seedTC(u) = mean(abs(yCC));
    L = length(ySeed);
    NFFT = 2^nextpow2(L); % Next power of 2 from length of y
    ySeedf = fft(ySeed,NFFT)/L;
    f = rsfc.fs/2*linspace(0,1,NFFT/2+1);
    ySeedunfiltf = fft(ySeedunfilt,NFFT)/L;
    ySeedunGRf = fft(ySeedunGR,NFFT)/L;
    filtandGRpower(u) = rsfc.df*sum(abs(ySeedf).^2);
    unfiltpower(u) = rsfc.df*sum(abs(ySeedunfiltf).^2);
    filteredpower(u) = rsfc.df*sum(abs(ySeedunGRf).^2);
    
    f2 = figure(202);
    subplot(5,1,1)
    plot(rsfc.tRS,ySeedunfilt);
    xlabel('time');
    ylabel('magnitude')
    title('Unfiltered Signal');
    subplot(5,1,2)
    plot(rsfc.tRS,rsfc.yglobal);
    xlabel('time');
    ylabel('magnitude')
    title('Global Unfiltered Signal');
    subplot(5,1,3)
    plot(rsfc.tRS,ySeed)
    xlabel('time');
    ylabel('magnitude');
    title('Filtered Signal');
    subplot(5,1,4)
    semilogx(f,2*abs(ySeedf(1:NFFT/2+1)),'b','LineWidth',2); 
    hold on
    xlim([0.005 1]);
    semilogx(f,2*abs(ySeedunfiltf(1:NFFT/2+1)),'r','LineWidth',2); 
    title('Frequency Spectrum');
    xlabel('frequency');
    ylabel('power');
    subplot(5,1,5)
    semilogx(f,2*abs(rsfc.meanyallf(1:NFFT/2+1)),'b','LineWidth',2); 
    xlim([0.005 1]);
    title('Golabl Frequency Spectrum');
    xlabel('frequency');
    ylabel('power'); 
    saveas(f2, [pathnameout '/Seed' num2str(u) 'plot.tif']);
    close(f2);
end
 
T = table(seedTC, filtandGRpower, unfiltpower,  filteredpower);
writetable(T,[pathnameout '/Outputdata.csv']);



function edit_filterunGR_Callback(hObject, eventdata, handles)
% hObject    handle to edit_filterunGR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_filterunGR as text
%        str2double(get(hObject,'String')) returns contents of edit_filterunGR as a double


% --- Executes during object creation, after setting all properties.
function edit_filterunGR_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_filterunGR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_selectInfarctRegion.
function pushbutton_selectInfarctRegion_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_selectInfarctRegion (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global rsfc

global rsfc
while(1)
  axes(handles.axes1)
  MinMinS = min(rsfc.I0(:)); MaxMaxS = max(rsfc.I0(:));
  while(1)
    imagesc(rsfc.I0(:,:,1),[ MinMinS MaxMaxS]); colorbar;
    colormap('gray');
    button = questdlg('Are you happy with the image contrast?','Repeat selection or not...','No');
    if strcmp(button,'No')
        answer= inputdlg({'Enter minimum intensity level of survey scan image'},'Enter minimum intensity level:',1,{num2str(MinMinS)});  %    {'0'});
        answer1 = cell2struct(answer, 'number', 1);
        MinMinS = str2num(answer1.number);
        answer= inputdlg({'Enter maximum intensity level of survey scan image'},'Enter maximum intensity level:',1,{num2str(MaxMaxS)});  %    {'0'});
        answer1 = cell2struct(answer, 'number', 1);
        MaxMaxS = str2num(answer1.number);
    else 
        break;
    end
  end
  rsfc.MaxS = MaxMaxS;
  rsfc.MinS = MinMinS;

  h = msgbox(' Please select the brain region');
  uiwait(h)
%   axes(handles.axes6)
%   imagesc(rsfc.yLRC,[-1 1]); 
  [brain_mask,Xi,Yi] = roipoly;
  hold on;
  plot(Xi,Yi,'color','k');
  hold off;
  button = questdlg('Are you statisfied with ROI?');
  if strcmp(button, 'Yes')
        break;
  end
end
  if (get(handles.radiobutton4,'Value') == get(handles.radiobutton4,'Max'))
    figure; imshow(rsfc.I0,[MinMinS MaxMaxS]); colormap('gray');
  end

% save('rsfc_brainMask.mat','brain_mask','Xi','Yi');
rsfc.infarct_mask = brain_mask;
% rsfc.Xi = Xi;
% rsfc.Yi = Yi;
