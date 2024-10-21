const express = require('express')
const router = express.Router()
const {createUser} = require('../Controllers/RegisterControllers')

router.post('/register', createUser )

module.exports = router;