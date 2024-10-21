const model = require('../Models/RegisterModel');

const createUser = async (req, res) => {
    const { name, email, password } = req.body

    try {
        const Create = await model.create({ name, email, password })
        res.status(200).json(Create)
    }
    catch (e) {
        res.status(400).json({ error: e.message });
    }
};

module.exports = { createUser }