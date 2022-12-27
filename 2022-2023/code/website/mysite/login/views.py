from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.auth.models import User


def index(request):
    return render(request, 'modelsite/home.html')

def reg(request):
    return render(request, 'modelsite/register.html')

def regsubmit(request):
    username = request.POST['username']
    password = request.POST['psw']
    invalid = False

    # password validation
    if len(password) < 3 or len(password) > 50:
        messages.add_message(request, messages.ERROR, 'Password should be between 3-50 characters')
        invalid = True

    # username validation
    if len(username) < 3 or len(username) > 50:
        messages.add_message(request, messages.ERROR, 'Username should be between 3-50 characters')
        invalid = True
    if invalid:
        return render(request, 'modelsite/register.html')

    # check if user exists and add if not
    try:
        u = User.objects.get(username = username)
        messages.add_message(request, messages.ERROR, 'Username already exists')
        return render(request, 'modelsite/register.html')

    except:
        user = User.objects.create_user(username=username, password=password)

    messages.add_message(request, messages.SUCCESS, 'Registration completed, please log in')

    return render(request, 'modelsite/login.html')


def loginpage(request):
    return render(request, 'modelsite/login.html')

def loginsubmit(request):

    username = request.POST['username']
    password = request.POST['psw']

    user = authenticate(request, username=username, password=password)

    # exists
    if user is not None:
        login(request, user)
        context = {"user": user}
        if user.is_superuser:
            
            return render(request, 'modelsite/adminHome.html', context)
        else:
            return render(request, 'modelsite/userHome.html', context)
    else:
        messages.add_message(request, messages.ERROR, 'Wrong username or password')
        return render(request, 'modelsite/login.html')



def userHome(request):
    return render(request, 'modelsite/userHome.html')

def adminHome(request):
    return render(request, 'modelsite/adminHome.html')

def viewUsers(request):

    context = {}
    context["users"] = User.objects.exclude(username = request.user.username)
    return render(request, 'modelsite/viewUsers.html', context=context)

def deleteUser(request, username):

    user = User.objects.get(username=username)
    user.delete()

    context = {}
    context["users"] = User.objects.exclude(username = request.user.username)

    messages.add_message(request, messages.SUCCESS, 'User deleted')

    return render(request, 'modelsite/viewUsers.html', context=context)