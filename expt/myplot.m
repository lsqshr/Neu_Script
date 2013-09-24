function myplot(x,y)

hplot = plot(x,y);
h = uicontrol('style','slider','units','pixel','position',[20 20 300 20]);
addlistener(h,'ActionEvent',@(hObject, event) makeplot(hObject, event,x,hplot));

function makeplot(hObject,event,x,hplot)
n = get(hObject,'Value');
set(hplot,'ydata',x.^n);
drawnow;