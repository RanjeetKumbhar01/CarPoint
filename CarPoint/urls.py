
from django.contrib import admin
from django.urls import path
from CarPoint import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path("",views.index,name='index'),
    path('buyingPrice/',views.buyingPrice,name='buy price'),
    path('sellingPrice/',views.sellingPrice,name='sell price'),
    path('model1/',views.firstModel,name='model 1'),
    path('model2/',views.secondModel,name='model 2'),
    path('model3/',views.thiredModel,name='model 3'),
    path('dataset1/',views.dataset1,name='dataset 1'),
    path('dataset2/',views.dataset2,name='dataset 2'),
    path('dataset3/',views.dataset3,name='dataset 3'),
    path('dataset4/',views.dataset4,name='dataset 4'),
    path('dataset5/',views.dataset5,name='dataset 5'),
    path('aboutUs/',views.aboutUs,name='about us'),
    path('discover/',views.discover,name='discover'),
    path('buy',views.buy,name='buy'),
    
]
