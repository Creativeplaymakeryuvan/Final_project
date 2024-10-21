const mongoose = require('mongoose');

const db = async () => {
    try{
        mongoose.set('strictQuery', false)
        await mongoose.connect('mongodb+srv://admin:admin@mern.cokdy.mongodb.net/?retryWrites=true&w=majority&appName=MERN')
    }
    catch (error){
        console.log('Db Connection Error')
    }
}

module.exports = {db}