#include <enph353_gazebo/collision_plugin.h>

#include <gazebo/common/Plugin.hh>
#include <gazebo/transport/transport.hh>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <gazebo/physics/physics.hh>

namespace gazebo
{

// Register plugin with gazebo
GZ_REGISTER_WORLD_PLUGIN(CollisionPlugin);

CollisionPlugin::CollisionPlugin() : WorldPlugin(), collisionNode(new transport::Node())
{
}

void CollisionPlugin::Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
{

  // Make sure the ROS node for Gazebo has already been initialized
  if (!ros::isInitialized())
  {
    ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
    << "Load the Gazebo system plugin 'libcollision_plugin.so' in the enph_ai package)");
    return;
  }

  this->world = _world;

  // Init time
  this->lastPubTime = this->world->SimTime();

  // Parse sdf period
  if (_sdf->HasElement("pubPeriod"))
    this->pubPeriod = _sdf->GetElement("pubPeriod")->Get<double>();
  else
    this->pubPeriod = 0.5;

  // Parse sdf topic
  if (_sdf->HasElement("pubTopic"))
    this->pubTopic = _sdf->GetElement("pubTopic")->Get<std::string>();
  else
    this->pubTopic = "isHit";

  // Parse sdf topic
  if (_sdf->HasElement("collisionObject1"))
    this->collisionObject1 = _sdf->GetElement("collisionObject1")->Get<std::string>();
  else
    this->collisionObject1 = "R1";

  // Setup collision node and sub
  ROS_INFO("Initializing collision plugin!");
  this->collisionNode->Init();
  this->collisionSub = this->collisionNode->Subscribe("/gazebo/default/physics/contacts",
                                          &CollisionPlugin::OnCollisionMsg, this);

  // Initialize ROS transport.
  this->rosNode.reset(new ros::NodeHandle());
  this->contactPub = this->rosNode->advertise<std_msgs::Bool>("/" + this->pubTopic, 100);

}

void CollisionPlugin::OnCollisionMsg(ConstContactsPtr &_contacts)
{
  int seconds = static_cast<int>(this->pubPeriod);
  int nanoSeconds = static_cast<int>((this->pubPeriod - seconds) * 1000000000);
  if (this->world->SimTime() - this->lastPubTime >= gazebo::common::Time(seconds, nanoSeconds))
  {
    bool isHit = false;
    // Check each collision. If it involves robot and wall, then it is hit
    for (unsigned int i = 0; i < _contacts->contact_size(); ++i) {
      std::string collisionStr1 = _contacts->contact(i).collision1();
      std::string collisionStr2 = _contacts->contact(i).collision2();
      isHit = isHit || ((collisionStr1.find(this->collisionObject1) != std::string::npos && collisionStr2.find("full_town") == std::string::npos) ||
                        (collisionStr1.find("full_town") == std::string::npos && collisionStr2.find(this->collisionObject1) != std::string::npos));
    }

    // publish a ROS MSG
    this->contactMsg.data = isHit;
    this->contactPub.publish(this->contactMsg);

    // Update pub time
    this->lastPubTime = this->world->SimTime();
  }
}

}  // namespace gazebo

