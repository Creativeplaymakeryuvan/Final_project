const User = require('../Models/RegisterModel')

exports.getName = async (req, res) => {
    try {
        const userId = req.headers['user-id'];
        if (!userId) {
            return res.status(400).json({ message: 'User ID is required' });
        }
        const user = await User.findOne({_id: userId});
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }
        res.status(200).json({name: user.username})
    }
    catch (error) {
        console.error('Error fetching user:', error);
        res.status(500).json({ message: 'Server Error' })
    }
}