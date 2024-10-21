const express = require('express')
const mongoose = require('mongoose')
const cors = require("cors")
const bodyParser = require('body-parser')
const User = require('./Models/RegisterModel')
const bcrypt = require('bcrypt')
const jwt = require('jsonwebtoken')
const {readdirSync} = require('fs')
const { db } = require('./Db/db')
const { Routes } = require('react-router-dom')

const SECRET_KEY = 'secretkey'

const app = express();

//middleware
app.use(express.json())
app.use(cors())

//DB connection
const server  = () => {
    db()
    app.listen('3001', ()=>{
        console.log('listening to port:')
    })
}

server()

//routes

readdirSync('./Routes').map((route) => app.use('/api/v1', require('./Routes/' + route)))

// CRUD(or)Routes

app.post('/register', async (req, res) => {
    try {
        const { email, username, password } = req.body
        let user = await User.findOne({ username })
        if (user) {
            return res.status(401).json({ error: 'Username Already Exists' })
        }
        else {
            const hashedPassword = await bcrypt.hash(password, 10)
            const newUser = new User({ email, username, password: hashedPassword })
            await newUser.save()
            res.status(201).json({ message: 'User Created Succesfully' })
        }
    } catch (error) {
        res.status(500).json({ error: 'Error in signup' })
    }
})

app.get('/register', async (req, res) => {
    try {
        const users = await User.find()
        res.status(201).json(users)
    } catch (error) {
        res.status(500).json({ error: 'Unable to get users' })
    }
})

app.post('/login', async (req, res) => {
    try {
        const { username, password } = req.body
        const user = await User.findOne({ username })
        if (!user) {
            return res.status(401).json({ error: 'Username Not Found' })
        }
        const isPasswordValid = await bcrypt.compare(password, user.password)
        if (!isPasswordValid) {
            return res.status(401).json({ error: 'Password is wrong' })
        }
        const token = jwt.sign({ userId: user._id }, SECRET_KEY, { expiresIn: '1hr' })
        res.json(user)
    } catch (error) {
        res.status(500).json({ error: 'Error logging in' })
    }
})





