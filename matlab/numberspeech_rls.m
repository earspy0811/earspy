%Feature Extraction for a sliding win
%configuration for phone on table : 9.65, 10

%filename='experiment_5_sample/numberspeech.csv';
%filename='experiment_5_sample/fsdddigit9t.csv';
%filename='experiment_5_sample/eardigitreal.csv';
%filename='experiment_5_sample/speechnew.csv';
%filename='experiment_5_sample/digitNoVibration_3.csv';
filename='experiment_5_sample/numberspeech200final.csv';
fnameString="test"; 
fnameNeumericStart=1; 

num = csvread(filename) ;
[r,c] = size(num) ;
timeValue=num(:,1); 
endVal=timeValue(end);
endVal=(endVal-10)*10;
disp(endVal);

%Delete unecessary information
num(2:2:end,:) = [] ;


regionCount=0;
consecutiveChecker=0;
consecutiveStart=0;
consecutiveEnd=0;
savedEnd=0;


zerocount=0;
onecount=0;
twocount=0;
threecount=0;
fourcount=0;
fivecount=0;
sixcount=0;
sevencount=0;
eightcount=0;
ninecount=0

%Fs = 1/mean(diff(num(:,1)));  
%y_highpass=highpass(num(:,4),20,Fs);
%num(:,4)=y_highpass;
%high pass filter
Fs = 1/mean(diff(num(:,1)));  
y_highpass=highpass(num(:,4),18,Fs);
%calculate_highpass=highpass(num(:,4),2,Fs);
num(:,4)=y_highpass;

global nextRow;

[numbers, strings, raw] = xlsread('acctest.xls');
nextRow = size(raw, 1);



calculate=num;

 %Delete rows for specific condition
  clowIndices = find(calculate(:,1)<10);
  calculate(clowIndices,:) = []; 

  chighIndices = find(calculate(:,1)>80);
  calculate(chighIndices,:) = [];
  
  mainZ=calculate(:,4) ;
  meanV=mean(mainZ);
  
  %disp(mainZ)



notInside=0;
startObserver=0;
startConsecutive=0;
appStart=0;
endConsecutive=0;
tempEnd=0;
appEnd=0;
endObserver=0;
regionCount=0;
errorCount=0;
errorState=0;  
largeCount=0;
smallCount=0;





start=10;
for x=1:endVal  
    iterate=num;
    %disp(iterate);
    start=start+0.1;
    initValue=start;
    final=start+0.1;

    %Delete rows for specific condition
    lowIndices = find(iterate(:,1)<initValue);
    iterate(lowIndices,:) = [];

    highIndices = find(iterate(:,1)>final);
     iterate(highIndices,:) = [];

    %disp(lowIndices);

    



    %Extract all axes
    ax = iterate(:,2) ;
    ay = iterate(:,3) ;
    az = iterate(:,4) ;
    %disp(az);

    meanX=mean(az);
    %disp(meanX);
    minX=min(az);
    maxX=max(az);
    
    %disp(start);
    %disp(min(az));
    %disp(max(az));
    
   
    try
     if min(az)<=-0.00275 || max(az)>=0.00275
            %disp(start);
            if(notInside==0 && startObserver==0)
                startObserver=1;
                tempStart=start;
                startConsecutive=1;        
            elseif(notInside==0 && startObserver==1)
                    startConsecutive=startConsecutive+1;
                    if(startConsecutive>=2)
                        notInside=1;
                        startObserver=0;
                        appStart=tempStart;
                        %fprintf('The starting Point is %d\n',appStart);
                        if(appStart-savedEnd)>0.5 && (appStart-errorState)>1.0
                            errorCount=errorCount+1;
                            %disp(appStart);
                            errorState=appStart;
                            fprintf('Possible Error before that %d\n',appStart);

                        end
                    end
            elseif(notInside==1 && endObserver==1)
                endObserver=0;
                tempEnd=0;
            end


        else
            if (notInside==1 && endObserver==0)
                endConsecutive=1;
                endObserver=1;
                tempEnd=final;
                %disp(tempEnd);
                %disp(tempEnd-appStart);
                if(tempEnd-appStart>0.4) 
                   %disp(tempEnd);
                   appEnd=tempEnd;
                    notInside=0; 
                end

            elseif (notInside==1 && endObserver==1)
                endConsecutive=endConsecutive+1;
                if(endConsecutive>=1)
                    if(tempEnd-appStart>0.3)
                        appEnd=tempEnd;
                        notInside=0;
                    end

                end
                if(endConsecutive>=2)
                    appEnd=tempEnd;
                    notInside=0;              
                end

            else
                if(notInside==0 && startObserver==1)
                    startObserver=0;
                    tempStart=0;
                end
            end  

        end 
    catch
        disp("Finished")
        break
    end
    
    
    %Now printing start and end
    
    if(appStart~=0 && appEnd~=0 && appStart<appEnd && appEnd-appStart>=0.3 && appEnd-appStart<=1.0)
        disp(appStart);
        disp(appEnd);
        disp("==========");
        
        if(appEnd-appStart)>=0.8
            largeCount=largeCount+1;
        elseif(appEnd-appStart)<=0.3
            smallCount=smallCount+1;
        end
        regionCount=regionCount+1;
        savedStart=appStart;
        savedEnd=appEnd;
        appStart=0;
        appEnd=0;


         %{
        if(savedStart<=95)
             class="zero";
             zerocount=zerocount+1;
        elseif(savedStart>95 && savedStart<=168)
            class="one";
            onecount=onecount+1;
        elseif(savedStart>168 && savedStart<=237)
            class="two";
            twocount=twocount+1;
        elseif(savedStart>237 && savedStart<=311)
            class="three";
            threecount=threecount+1;
        elseif(savedStart>311 && savedStart<=382)
            class="four";
            fourcount=fourcount+1;
        elseif(savedStart>382 && savedStart<=456)
            class="five";
            fivecount=fivecount+1;
        elseif(savedStart>456 && savedStart<=541)
            class="six";
            sixcount=sixcount+1;
        elseif(savedStart>541 && savedStart<=617)
            class="seven";
            sevencount=sevencount+1;
        elseif(savedStart>617 && savedStart<=688)
            class="eight";
            eightcount=eightcount+1;
        elseif(savedStart>688)
            class="nine";
            ninecount=ninecount+1;
        end

        %}

        
        
        
        
        
        
        
        if(savedStart<=120)
             class="zero";
             zerocount=zerocount+1;
        elseif(savedStart>120 && savedStart<=222)
            class="one";
            onecount=onecount+1;
        elseif(savedStart>222 && savedStart<=318)
            class="two";
            twocount=twocount+1;
        elseif(savedStart>318 && savedStart<=420)
            class="three";
            threecount=threecount+1;
        elseif(savedStart>420 && savedStart<=519)
            class="four";
            fourcount=fourcount+1;
        elseif(savedStart>519 && savedStart<=621)
            class="five";
            fivecount=fivecount+1;
        elseif(savedStart>621 && savedStart<=735)
            class="six";
            sixcount=sixcount+1;
        elseif(savedStart>735 && savedStart<=840)
            class="seven";
            sevencount=sevencount+1;
        elseif(savedStart>840 && savedStart<=937)
            class="eight";
            eightcount=eightcount+1;
        elseif(savedStart>937)
            class="nine";
            ninecount=ninecount+1;
        end

      
        
 
          my_xls(filename,savedStart,savedEnd,endVal,class);
    end
    
end
fprintf('Total word region found %d\n',regionCount);
fprintf('Total possible error found %d\n',errorCount);
fprintf('Total oversized word regions %d\n',largeCount);
fprintf('Total undersized word regions %d\n',smallCount);

fprintf('Zero: %d\n',zerocount);
fprintf('One: %d\n',onecount);
fprintf('Two: %d\n',twocount);
fprintf('Three %d\n',threecount);
fprintf('Four: %d\n',fourcount);
fprintf('Five: %d\n',fivecount);
fprintf('Six: %d\n',sixcount);
fprintf('Seven %d\n',sevencount);
fprintf('Eight: %d\n',eightcount);
fprintf('Nine: %d\n',ninecount);



function my_xls(filename,start,funcend,endVal,class)


startValue=start;
endValue=funcend;
%watchCompareStart=20;
%watchCompareEnd=34;
phoneCompareStart=10;
phoneCompareEnd=endVal+10;
%disp(watchStartValue);
%disp(endValue);
%disp(watchEndValue);
global nextRow;

%now starting the smartphonw calculations

num3 = csvread('experiment_5_sample/emodbhang.csv');

numPhone = csvread(filename) ;
[r1,c1] = size(numPhone) ;

numPhone(2:2:end,:) = [] ;
num3(2:2:end, :) = [];



comparePhone=numPhone;

%Delete rows for phone compare values
comparePLow = find(comparePhone(:,1)<phoneCompareStart);
comparePhone(comparePLow,:) = [];

comparePHigh = find(comparePhone(:,1)>phoneCompareEnd);
comparePhone(comparePHigh,:) = [];



Fsp = 1/mean(diff(numPhone(:,1)));  
Fn=Fsp/2;

%experimenting highpass
%y_highpass=highpass(numPhone(:,4),15,Fsp);
%numPhone(:,4)=y_highpass;

%y_lowpass=lowpass(numPhone(:,4),4,Fsp);
%numPhone(:,4)=y_lowpass;


%RLS 

% Extract the Z-axis signals
signal = numPhone(:, 4); % Noisy signal
reference_signal = num3(:, 4); % Clean reference signal

% Ensure both signals are of the same length
min_length = min(length(signal), length(reference_signal));
signal = signal(1:min_length);
reference_signal = reference_signal(1:min_length);

% Sampling frequency
Fs = 1 / mean(diff(numPhone(:, 1))); % Sampling frequency (adjust as needed)
disp(['Sampling Frequency: ', num2str(Fs), ' Hz']);

% Step 1: Perform Adaptive Noise Cancellation using RLS Filtering
filter_order = 32; % Filter order
forgetting_factor = 0.97; % Forgetting factor for RLS filter

% Initialize RLS filter variables
weights = zeros(filter_order, 1); % Initial filter weights
P_matrix = eye(filter_order) * 1000; % Initial inverse correlation matrix (large value)
cleaned_signal_rls = zeros(size(signal)); % Output signal

% Perform RLS adaptive filtering
for n = filter_order:length(signal)
    input_vector = signal(n:-1:n-filter_order+1); % Input vector (reversed)
    k_vector = (P_matrix * input_vector) / ...
               (forgetting_factor + input_vector' * P_matrix * input_vector); % Gain vector
    error_n = reference_signal(n) - weights' * input_vector; % Error signal
    weights = weights + k_vector * error_n; % Update filter weights
    P_matrix = (P_matrix - k_vector * input_vector' * P_matrix) / forgetting_factor; % Update P matrix
    cleaned_signal_rls(n) = weights' * input_vector; % Store the cleaned signal
end


numPhone(:,4)=cleaned_signal_rls;

%Delete rows for specific condition in Phone
lowIndicesp = find(numPhone(:,1)<startValue);
numPhone(lowIndicesp,:) = [];

highIndicesp = find(numPhone(:,1)>endValue);
numPhone(highIndicesp,:) = [];

phoneRms = numPhone(:,5) ;

phoneZ=numPhone(:,4);
comparePZ=comparePhone(:,4);

%Calculating Frequency domain features

Tr = linspace(numPhone(1,1), numPhone(1,end), size(numPhone,1));  
Dr = resample(phoneZ, Tr); 
Dr_mc  = Dr - mean(Dr,1); 


FDr_mc = fft(Dr_mc, [], 1);
Fv = linspace(0, 1, fix(size(FDr_mc,1)/2)+1)*Fn; 

Iv = 1:numel(Fv); 
amplitude=abs(FDr_mc(Iv,:))*2;

upperPart=Fv*amplitude;
ampSum=sum(amplitude);

specCentroid=upperPart/ampSum;
%disp(specCentroid); 

FvSqr=Fv.^2;
stdDevupper=FvSqr*amplitude;
specStdDev=sqrt(stdDevupper/ampSum);
specCrest=max(amplitude)/specCentroid;


specSkewness=(((Fv-specCentroid).^3)*amplitude)/(specStdDev)^3;

specKurt=(sum((((amplitude-specCentroid).^4).*amplitude))/(specStdDev)^4)-3 ;
maxFreq=max(Fv);
maxMagx=max(phoneZ);







meanP=mean(phoneZ);
minP=min(phoneZ);
maxP=max(phoneZ);
meanPZ=mean(comparePZ);
%gradientZ=mean(gradient(phoneZ));
%disp(meanPZ); 
irrk=irregularityk(phoneZ);
irrj=irregularityj(phoneZ);
sharp=sharpness(phoneZ);
smooth=smoothness(phoneZ);

%now adding frequency domain things:





%disp(meanP);
%disp(minP);
%disp(maxP);

meanCrossingP=phoneZ > meanPZ;
numberCrossingP=sum(meanCrossingP(:) == 1);
meanCrossingRateP=numberCrossingP/numel(phoneZ);
%disp(meanCrossingRateP);



%Extracting frequency domain values:

Fp = fft(phoneZ,1024);
FFTCoEffp=Fp/length(phoneZ);
powp = Fp.*conj(Fp);
total_powp = sum(powp);
%disp(total_powp);



Fsp = 1/mean(diff(numPhone(:,1)));

penp=pentropy(phoneZ,Fsp);
sumPenp=sum(penp);

%disp(sumPenp);

%centroid=spectralCentroid(phoneZ,Fsp);
%disp(centroid)

%sharpness = acousticSharpness(phoneZ,Fsp);
%disp(sharpness);



hdp = dfilt.fftfir(phoneZ,1024);
cp=fftcoeffs(hdp);
ampp = 2*abs(cp)/length(phoneZ);
phasep=angle(cp);
magnitudep=abs(ampp);

highestMagp=max(magnitudep);
sumMagp=sum(magnitudep);

frequency_ratiop=highestMagp/sumMagp;



statX=[mean(phoneZ) max(phoneZ) min(phoneZ) std(phoneZ) var(phoneZ) range(phoneZ) (std(phoneZ)/mean(phoneZ))*100 skewness(phoneZ) kurtosis(phoneZ) quantile(phoneZ,[0.25,0.50,0.75]) meanCrossingRateP total_powp sumPenp frequency_ratiop irrk irrj sharp smooth specCentroid specStdDev specCrest specSkewness specKurt maxFreq maxMagx class];

t=array2table(statX);

nextRow = nextRow + 1;
cellReference = sprintf('A%d', nextRow);
xlswrite('acctest4.xls', statX, 'Sheet1', cellReference);









end


function ikSum=irregularityk(phonez)
%disp(phonez);
N=[10 100 1000];

ikSum=0;
for i=1:length(phonez)-2
   ik=(phonez(i+1)-(phonez(i)+phonez(i+1)+phonez(i+2)/3));
   ikSum=ikSum+ik;
    
end
end

function ijSum=irregularityj(phonez)


ijSum=0;
for i=1:length(phonez)-1
   ij1=(phonez(i)-phonez(i+1))^2;
   ij2=phonez(i)^2;
   ij=ij1/ij2;
   ijSum=ijSum+ij;
    
end
end

function finalsharp=sharpness(phonez)
sharpn=0;
tempi=0;
for i=1:length(phonez)
    if(i<15)
        tempi=real(i*phonez(i)^0.23);
        %disp(tempi);
    else
        tempi=real(0.066*exp(0.171*i)*i*phonez(i)^0.23);
    end
    
    sharpn=sharpn+tempi;
end

finalsharp=(0.11*sharpn)/length(phonez);
end

function smoothSum=smoothness(phonez)

smoothSum=0;
for i=1:length(phonez)-2
   
    ismooth=real((20*log(phonez(i))-(20*log(phonez(i))+20*log(phonez(i+1))+20*log(phonez(i+2))))/3);
    
    smoothSum=smoothSum+ismooth;
end


end








