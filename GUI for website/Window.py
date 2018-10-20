# -*- coding: utf-8 -*-
from Tkinter import *
import Tkinter as tk
from tkMessageBox import *
import tkFileDialog

import requests as req
import os

class Window:
	
	def __init__(this):
		# SERVER -----------------------
		this.server = "http://localhost:12000" # Change this

		this.username = "admin" # Change this
		this.password = "12345" # Change this 

		# WINDOW -----------------------
		this.window = Tk()
		this.window.title("Recipes")
		# this.window.iconbitmap("favicon.ico")
		this.window.resizable(width=False, height=False)
		this.window.protocol("WM_DELETE_WINDOW", this.window.destroy)

		this.windowWidth = 1000
		this.windowHeight = 800
		this.window.geometry("%sx%s" % (this.windowWidth, this.windowHeight))

		this.textHeight = 21






		# Main header frame
		this.headerFrame = Frame(this.window, height=1)
		this.headerFrame.grid(row=0, column=0)








		# Frame, Labels and dropdown menu (Methods)

		# Frame ---------------
		this.methodsFrame = Frame(this.headerFrame, height=1)
		this.methodsFrame.grid(row=0, column=0)

		# label -----------------
		this.methodsLabel = Label(this.methodsFrame, text="Choose an option:")
		this.methodsLabel.grid(row=0, column=0)

		# Dropdown menu ------------------
		this.methodsDropStr = StringVar(this.methodsFrame)
		this.methods = ["Add a recipe", "Remove a recipe", "Change a recipe"]
		this.methodsDropStr.set(this.methods[0])
		this.methodsDropStr.trace("w", this.methodsDropChanged)

		this.methodsDrop = OptionMenu(this.methodsFrame, this.methodsDropStr, *this.methods)
		this.methodsDrop.grid(row=0, column=1)







		# Frame, Labels and dropdown menu (Category)

		# Frame -----------------
		this.catFrame = Frame(this.headerFrame, height=1)
		this.catFrame.grid(row=0, column=1)

		# Label -----------------
		this.catLabel = Label(this.catFrame, text="Choose a category:")
		this.catLabel.grid(row=0, column=0)

		# Dropdown menu ---------------------
		this.catDropStr = StringVar(this.catFrame)
		this.categories = this.getCategories()
		this.catDropStr.set(this.categories[0])
		this.catDropStr.trace("w", this.catDropChanged)

		this.catDrop = OptionMenu(this.catFrame, this.catDropStr, *this.categories)
		this.catDrop.grid(row=0, column=1)










		# Frame, Labels, Text and dropdown menu (Meal)
		
		# Frame --------------
		this.mealFrame = Frame(this.headerFrame, height=1)
		this.mealFrame.grid(row=0, column=2)

		# Label --------------------
		this.mealLabel = Label(this.mealFrame, text="Write the name of the meal:")
		this.mealLabel.grid(row=0, column=0)

		# Text ---------------------
		this.mealText = Text(this.mealFrame, width=30, height=1)
		this.mealText.grid(row=0, column=1)
		this.mealText.bind("<KeyRelease>", this.textChanged)

		# Dropdown menu --------------------
		this.mealDropStr = StringVar(this.mealFrame)
		this.meals = this.getMeals(this.catDropStr.get())
		this.mealDropStr.set(this.meals[0])
		this.mealDropStr.trace("w", this.mealDropChanged)

		this.mealDrop = OptionMenu(this.mealFrame, this.mealDropStr, *this.meals)
		this.mealDrop.grid_forget()



		# Frame, Labels and Text (Ingredients)

		# Frame ----------------
		this.ingrFrame = Frame(this.window)
		this.ingrFrame.grid(row=1, column=0, sticky="nswe")

		this.window.grid_rowconfigure(1, weight=1)
		this.window.grid_columnconfigure(0, weight=1)

		# Label -----------------
		this.ingrLabel = Label(this.ingrFrame, text="Write the ingredients:")
		this.ingrLabel.grid(row=0, column=0)

		# Text -------------------
		this.ingrText = Text(this.ingrFrame, height=this.textHeight)

		this.ingrText.grid(row=1, column=0, sticky="we")
		this.ingrFrame.grid_rowconfigure(1, weight=1)
		this.ingrFrame.grid_columnconfigure(0, weight=1)

		

		




		# Frame, Labels and Text (Steps)

		# Frame ----------------		
		this.stepsFrame = Frame(this.window,)
		this.stepsFrame.grid(row=2, column=0, sticky="nswe")

		this.window.grid_rowconfigure(2, weight=1)
		this.window.grid_columnconfigure(0, weight=1)

		# Label ----------------
		this.stepsLabel = Label(this.stepsFrame, text="Write the steps:")
		this.stepsLabel.grid(row=0, column=0)

		# Text ----------------
		this.stepsText = Text(this.stepsFrame, height=this.textHeight)

		this.stepsText.grid(row=1, column=0, sticky="we")
		this.stepsFrame.grid_rowconfigure(1, weight=1)
		this.stepsFrame.grid_columnconfigure(0, weight=1)








		# Frame, Buttons (Accept, Clear)

		# Frame ------------------
		this.buttonsFrame = Frame(this.window, bg="Black", height=1)
		this.buttonsFrame.grid(row=3, column=0)

		this.window.grid_rowconfigure(3, weight=1)
		this.window.grid_columnconfigure(0, weight=1)

		this.acceptButton = Button(this.buttonsFrame, text="Accept", command=this.sendData)
		this.acceptButton.grid(row=0, column=0)

		this.clearButton = Button(this.buttonsFrame, text="Clear", command=this.clearTexts)
		this.clearButton.grid(row=0, column=1)

		this.imageButton = Button(this.buttonsFrame, text="Select image", command=this.selectImage)
		this.imageButton.grid(row=0, column=2)
		this.image = ""
		this.imageExt = ""


	# Selection the image
	def selectImage(this, *args):
		curDir = os.getcwd()

		this.image = tkFileDialog.askopenfilename(title="Select an image", initialdir=curDir, filetypes = (("jpg files", "*.jpg"), ("png files", "*.png")))
		
		if(this.image != ""):
			this.imageButton.config(text="Image is selected")
			this.imageExt = this.image.split(".")[-1]
			this.image = open(this.image, "rb").read().encode("base64").replace("\n", "")
		else:
			this.imageButton.config(text="Select image")

	# Text widget contents are changed
	def textChanged(this, *args):
		
		str = this.mealText.get(1.0, "end")

		if "\n" in str:
			str = str.replace("\n", "")
			this.mealText.delete(1.0, "end")
			this.mealText.insert(1.0, str)

	# Methods option menu is changed
	def methodsDropChanged(this, *args):
	
		if(this.methodsDropStr.get() == "Change a recipe"):

			this.mealLabel.config(text="Choose a meal")

			this.mealDrop.grid(row=0, column=1)
			this.mealText.grid_forget()

			if(this.catDropStr.get() != "Add a new category..."):
				this.meals = this.getMeals(this.catDropStr.get())

				this.mealDrop["menu"].delete(0, "end")
				this.mealDropStr.set(this.meals[0])
				for choice in this.meals:
					this.mealDrop["menu"].add_command(label=choice, command=tk._setit(this.mealDropStr, choice))

		elif(this.methodsDropStr.get() == "Remove a recipe"):

			this.clearTexts()

		else:

			this.mealLabel.config(text="Write the name of the meal:")

			this.mealDrop.grid_forget()
			this.mealText.grid(row=0, column=1)

			this.clearTexts()

	# Get the categories from the server
	def getCategories(this):

		url = this.server + "/getCategories"
		res = req.post(url).content.split(",")
		
		res.append("Add a new category...")
		
		return res

	# Get the meals of a category from the server
	def getMeals(this, category):

		data = {"Category": category}

		url = this.server + "/getMeals"
		res = req.post(url, json=data).content.split(",")

		return res

	# Category option menu is changed
	def catDropChanged(this, *args):

		if(this.catDropStr.get() == "Add a new category..."):
			this.catPop()
		elif(this.methodsDropStr.get() == "Change a recipe"):
			this.meals = this.getMeals(this.catDropStr.get())

			this.mealDrop["menu"].delete(0, "end")
			this.mealDropStr.set(this.meals[0])
			for choice in this.meals:
				this.mealDrop["menu"].add_command(label=choice, command=tk._setit(this.mealDropStr, choice))

		print("Drop menu changed to %s" % this.catDropStr.get())

	# Pop up window for adding a new category
	def catPop(this):
		popup = Toplevel()
		popup.title("Adding a new category")
		# popup.iconbitmap("favicon.ico")
		popup.protocol("WM_DELETE_WINDOW", popup.destroy)

		mainFrame = Frame(popup, height=1)
		mainFrame.grid(row=0, column=0)

		label = Label(mainFrame, text="Write a new category:")
		label.grid(row=0, column=0)		

		input = Entry(mainFrame)
		input.grid(row=0, column=1)

		buttonFrame = Frame(popup, height=1)
		buttonFrame.grid(row=1, column=0)

		doneButton = Button(buttonFrame, text="Done", command= lambda event=None, 
																	  popup=popup, 
																	  inputCat=input: this.popDone(popup, input))
		doneButton.grid(row=0, column=0)

		cancelButton = Button(buttonFrame, text="Cancel", command=popup.destroy)
		cancelButton.grid(row=0, column=1)

	# When the user clicks the done button
	def popDone(this, popup, input):
		input = input.get().rstrip()

		if(input.replace(" ", "") != ""):
			lastEl = "Add a new category..."
			this.categories[-1] = input
			this.categories.append(lastEl)


			this.catDrop["menu"].delete(0, "end")
			this.catDropStr.set(input)
			for choice in this.categories:
				this.catDrop["menu"].add_command(label=choice, command=tk._setit(this.catDropStr, choice))
		

		popup.destroy()
		
	# Meal option menu is changed
	def mealDropChanged(this, *args):
		data = {"Category": this.catDropStr.get(), "Food": this.mealDropStr.get()}

		url = this.server + "/getRecipe"
		recipe = req.post(url, json=data).content.split(",")

		this.ingrText.delete(1.0, "end")
		this.ingrText.insert(1.0, recipe[0])

		this.stepsText.delete(1.0, "end")
		this.stepsText.insert(1.0, recipe[1])

	# Renders the window
	def render(this):
		this.window.mainloop()

	# Formats and sends the data to the server
	def sendData(this, *args):
		method = this.methodsDropStr.get()
		cat = this.catDropStr.get()
		meal = this.mealText.get(1.0, "end")
		mealStr = this.mealDropStr.get()
		ingr = this.ingrText.get(1.0, "end")
		steps = this.stepsText.get(1.0, "end")

		if(method == "Remove a recipe"):

			if(meal.replace(" ", "").rstrip() == ""):
				showerror(title="White texts boxes", message="There is 1 or more texts boxes that are empty")

			data = {"username": this.username.encode("base64"), "password": this.password.encode("base64"), "Category": cat, "Food": meal.rstrip()}

			try:
				res = req.post(this.server + "/remove", json=data)
			
				if(res.content == "Invalid credentials"):
					showerror(title="Invalid credentials", message="Wrong authentication.")

			except:
				showerror(title="Error", message="An unexpected error happened. \nContact your administrator")	


		elif(method == "Add a recipe"):

			if(meal.replace(" ", "").rstrip() == "" or ingr.replace(" ", "").rstrip() == "" or steps.replace(" ", "").rstrip() == ""):
				showerror(title="White texts boxes", message="There is 1 or more texts boxes that are empty")

			ingr = "<h3 class='contextTextHeader'>Ingredients:</h3>" + ingr[:-2]
			steps = "<h3 class'contextTextHeader'>Steps:</h3>" + steps

			data = {"username": this.username.encode("base64"), "password": this.password.encode("base64"), "Category": cat, "Food": meal.rstrip(), "Ingredients": ingr, "Steps": steps}

			if(this.image != ""):
				data["Image"] = this.image
				data["ImageExt"] = this.imageExt

			try:
				res = req.post(this.server + "/add", json=data)

				if(res.content == "Invalid credentials"):
					showerror(title="Invalid credentials", message="Wrong authentication.")

			except:
				showerror(title="Error", message="An unexpected error happened. \nContact your administrator")

		elif(method == "Change a recipe"):

			if(ingr.replace(" ", "").rstrip() == "" or steps.replace(" ", "").rstrip() == ""):
				showerror(title="White texts boxes", message="There is 1 or more texts boxes that are empty")

			while(True):
				if(ingr[-2:] == "\n"):
					ingr = ingr[:-2]
				else:
					break

			ingr = "<h3 class='contextTextHeader'>Ingredients:</h3>" + ingr
			steps = "<h3 class'contextTextHeader'>Steps:</h3>" + steps

			data = {"username": this.username.encode("base64"), "password": this.password.encode("base64"), "Category": cat, "Food": mealStr, "Ingredients": ingr, "Steps": steps}

			try:
				res = req.post(this.server + "/modify", json=data)

				if(res.content == "Invalid credentials"):
					showerror(title="Invalid credentials", message="Wrong authentication.")

			except:
				showerror(title="Error", message="An unexpected error happened. \nContact your administrator")

		this.clearTexts()

	# Clears all the texts widgets
	def clearTexts(this):
		this.mealText.delete(1.0, "end")
		this.ingrText.delete(1.0, "end")
		this.stepsText.delete(1.0, "end")

		this.imageButton.config(text="Select image");
		this.image = this.imageExt = ""