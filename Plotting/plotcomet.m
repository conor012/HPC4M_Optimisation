M = readmatrix('trajectory.csv');
Mbfgs = readmatrix('bfgstrajectory.csv');
x = M(:,1);
y = M(:,2);
xbfgs = Mbfgs(:,1);
ybfgs = Mbfgs(:,2);
X = -5:0.01:5;
Y = -5:0.01:5;
[X,Y] = meshgrid(X,Y);
surf(X,Y, X.^2+Y.^2,'EdgeColor','none','FaceAlpha',0.5);
hold on
% comet3(x,y,x.^2 + y.^2)
z = x.^2 + y.^2;
zbfgs = xbfgs.^2 + ybfgs.^2;
%%Initialize video
myVideo = VideoWriter('Quadratic.mp4','MPEG-4'); %open video file
myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
open(myVideo)
%% Plot in a loop and grab frames
for i=1:length(x)
if(abs(z(i) - min(z)) > 1e-3 && i<1e2)
    plot3(x(1:i), y(1:i), z(1:i),'r', 'LineWidth', 2);
    plot3(xbfgs(1:min(i,length(xbfgs))), ybfgs(1:min(i,length(xbfgs))), zbfgs(1:min(i,length(xbfgs))),'b', 'LineWidth', 2);
%     pause(0.01) %Pause and grab frame
    frame = getframe(gcf); %get frame
    writeVideo(myVideo, frame);
end
end
close(myVideo)
