const mongoose = require('mongoose')

const Schema = mongoose.Schema;

const RegisterSchema = new Schema(
    {
        email: { type: String, required: true },
        username: { type: String, required: true, unique: true},
        password: { type: String , required: true },
        incomes: [{
            type: mongoose.Schema.Types.ObjectId,ref: 'Income'
        }]
    },{ timestamps: true }
);
const User = mongoose.model("User",RegisterSchema)
module.exports = User