M = readmatrix('EggholderBFGStrajectory.csv');
% Mbfgs = readmatrix('bfgsEggtrajectory.csv');
x = M(:,1);
y = M(:,2);
% xbfgs = Mbfgs(:,1);
% ybfgs = Mbfgs(:,2);
% X = 0:0.5:512;
% Y = 0:0.5:512;
X = -512:0.5:512;
Y = -512:0.5:512;
[X,Y] = meshgrid(X,Y);

Z = egg(X, Y);
fig = figure('WindowState', 'maximized');
contour(X,Y,Z);

hold on
% axis square
% xlim([0 512])
% ylim([0 512])
xlim([min(min(X)), max(max(X))])
ylim([min(min(Y)), max(max(Y))])

% comet3(x,y,x.^2 + y.^2)
% z = egg(x, y);
z = egg(x,y);
% zbfgs = egg(xbfgs, ybfgs);

%%Initialize video
myVideo = VideoWriter('EggholderBFGS.mp4','MPEG-4'); %open video file
myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
open(myVideo)
%% Plot in a loop and grab frames
for i=1:length(x)
    if(abs(z(i) - min(z)) > 1e-4)
        plot(x(i), y(i),'r.', 'LineWidth', 1);
        title(['Iteration  ' num2str(i), '  Position   x = ', num2str(x(i)), '  y = ' , num2str(y(i))])
%         plot(xbfgs(1:min(i,length(xbfgs))), ybfgs(1:min(i,length(xbfgs))),'b', 'LineWidth', 2);
        pause(0.00000001) %Pause and grab frame
        frame = getframe(gcf); %get frame
        writeVideo(myVideo, frame);

    end
end
close(myVideo)

function z = egg(x,y)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% EGGHOLDER FUNCTION
%
% Authors: Sonja Surjanovic, Simon Fraser University
%          Derek Bingham, Simon Fraser University
% Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
%
% Copyright 2013. Derek Bingham, Simon Fraser University.
%
% THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
% FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
% derivative works, such modified software should be clearly marked.
% Additionally, this program is free software; you can redistribute it 
% and/or modify it under the terms of the GNU General Public License as 
% published by the Free Software Foundation; version 2.0 of the License. 
% Accordingly, this program is distributed in the hope that it will be 
% useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
% of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
% General Public License for more details.
%
% For function details and reference information, see:
% http://www.sfu.ca/~ssurjano/
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% INPUT:
%
% xx = [x1, x2]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


term1 = -(y+47) .* sin(sqrt(abs(y+x/2+47)));
term2 = -x .* sin(sqrt(abs(x-(y+47))));

z = term1 + term2;

end
function z = quadratic(x,y)

    z = x.^2 + y.^2;

end
